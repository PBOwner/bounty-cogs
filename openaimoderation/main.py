import asyncio
import base64
import logging
import re
from typing import Optional

import aiohttp
import discord
from redbot.core import commands
from redbot.core.bot import Red

log = logging.getLogger("red.cray-cogs.openaimoderation")

URL_REGEX = re.compile(r"https?://\S+")


class OpenAIModeration(commands.Cog):
    """
    Demos the OpenAI Content Moderation API for images.
    """

    __author__ = "Antigravity"
    __version__ = "1.0.0"

    def __init__(self, bot: Red):
        self.bot = bot
        self.session = aiohttp.ClientSession()

    async def cog_unload(self) -> None:
        await self.session.close()

    def detect_mime_type(self, image_bytes: bytes) -> Optional[str]:
        """Detects the MIME type of an image based on its magic numbers."""
        if len(image_bytes) < 4:
            return None
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if image_bytes.startswith(b"GIF8"):
            return "image/gif"
        if len(image_bytes) >= 12 and image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        return None

    async def download_image(self, url: str) -> tuple[bytes, str]:
        """Downloads an image from a URL and validates it."""
        try:
            async with self.session.get(url, timeout=15) as response:
                if response.status != 200:
                    raise commands.BadArgument(
                        f"Failed to download image from URL (HTTP status: {response.status})."
                    )

                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > 20 * 1024 * 1024:
                    raise commands.BadArgument(
                        "The image is too large. The OpenAI Moderation API limits images to 20MB."
                    )

                image_bytes = await response.read()

                if len(image_bytes) > 20 * 1024 * 1024:
                    raise commands.BadArgument(
                        "The image is too large. The OpenAI Moderation API limits images to 20MB."
                    )

                mime_type = response.headers.get("Content-Type", "")
                detected_mime = self.detect_mime_type(image_bytes)

                if detected_mime:
                    return image_bytes, detected_mime
                
                # If magic bytes detection failed, verify if the header indicates a supported type
                if mime_type in ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"]:
                    return image_bytes, mime_type

                raise commands.BadArgument(
                    "The provided URL does not point to a supported image format (PNG, JPEG, WEBP, GIF)."
                )
        except aiohttp.ClientError as e:
            log.error(f"Error downloading image from {url}: {e}", exc_info=True)
            raise commands.BadArgument(
                "An error occurred while downloading the image. Please verify the URL is valid and accessible."
            )
        except asyncio.TimeoutError:
            raise commands.BadArgument("Connection timed out while trying to download the image.")

    async def get_image_from_context(
        self, ctx: commands.Context, url: Optional[str]
    ) -> tuple[bytes, str, str]:
        """
        Extracts image bytes, MIME type, and source URL from context parameters.
        Checks url argument, attachments, and referenced message (replies).
        """
        # Case 1: URL argument provided
        if url:
            if not url.startswith(("http://", "https://")):
                raise commands.BadArgument(
                    "Invalid URL format. Please provide a URL starting with http:// or https://"
                )
            image_bytes, mime_type = await self.download_image(url)
            return image_bytes, mime_type, url

        # Case 2: Direct message attachments
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            if attachment.size > 20 * 1024 * 1024:
                raise commands.BadArgument(
                    "The attached image is too large. The OpenAI Moderation API limits images to 20MB."
                )
            image_bytes = await attachment.read()
            detected_mime = self.detect_mime_type(image_bytes)
            if not detected_mime:
                raise commands.BadArgument(
                    "The attached file is not a supported image format (PNG, JPEG, WEBP, GIF)."
                )
            return image_bytes, detected_mime, attachment.url

        # Case 3: Reply reference message
        if ctx.message.reference:
            ref = ctx.message.reference
            ref_msg = ref.resolved
            if not isinstance(ref_msg, discord.Message):
                try:
                    ref_msg = await ctx.channel.fetch_message(ref.message_id)
                except discord.HTTPException:
                    ref_msg = None

            if ref_msg:
                if ref_msg.attachments:
                    attachment = ref_msg.attachments[0]
                    if attachment.size > 20 * 1024 * 1024:
                        raise commands.BadArgument(
                            "The image in the referenced message is too large (> 20MB)."
                        )
                    image_bytes = await attachment.read()
                    detected_mime = self.detect_mime_type(image_bytes)
                    if not detected_mime:
                        raise commands.BadArgument(
                            "The referenced attachment is not a supported image format."
                        )
                    return image_bytes, detected_mime, attachment.url

                urls = URL_REGEX.findall(ref_msg.content)
                if urls:
                    image_bytes, mime_type = await self.download_image(urls[0])
                    return image_bytes, mime_type, urls[0]

        raise commands.BadArgument(
            "No image found. Please provide an image URL, attach an image, or reply to a message containing one."
        )

    async def query_openai_moderation(
        self, api_key: str, image_bytes: bytes, mime_type: str
    ) -> dict:
        """Queries the OpenAI Moderation API using the omni-moderation-latest model."""
        url = "https://api.openai.com/v1/moderations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # Encode image in base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_image}"

        payload = {
            "model": "omni-moderation-latest",
            "input": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ],
        }

        try:
            async with self.session.post(
                url, headers=headers, json=payload, timeout=30
            ) as response:
                if response.status == 401:
                    raise commands.BadArgument(
                        "Unauthorized: The OpenAI API key is invalid or inactive."
                    )
                elif response.status == 429:
                    raise commands.BadArgument(
                        "Rate Limited: Too many requests sent to the OpenAI Moderation API. Please try again later."
                    )
                elif response.status != 200:
                    try:
                        error_json = await response.json()
                        error_msg = error_json.get("error", {}).get(
                            "message", "Unknown error"
                        )
                    except Exception:
                        error_msg = await response.text()
                    raise commands.BadArgument(
                        f"OpenAI API Error (Status {response.status}): {error_msg}"
                    )

                return await response.json()
        except aiohttp.ClientError as e:
            log.error(f"HTTP client error during OpenAI Moderation: {e}", exc_info=True)
            raise commands.BadArgument(
                "A connection error occurred while communicating with the OpenAI API."
            )
        except asyncio.TimeoutError:
            raise commands.BadArgument("The request to the OpenAI Moderation API timed out.")

    @commands.command(name="moderateimage", aliases=["modimage", "imgmod", "modimg"])
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def moderate_image(self, ctx: commands.Context, url: Optional[str] = None):
        """
        Moderate an image using OpenAI's Content Moderation API.

        You can:
        1. Provide a direct image URL: `[p]moderateimage https://example.com/image.jpg`
        2. Upload an image attachment with the command.
        3. Reply to a message that contains an image or image URL.
        """
        # Fetch the shared API tokens
        apikeys = await self.bot.get_shared_api_tokens("openai")
        api_key = apikeys.get("api_key")

        if not api_key:
            await ctx.send(
                "⚠️ **The OpenAI API key is not set.**\n"
                "Please configure it first using the following bot command:\n"
                f"`{ctx.clean_prefix}set api openai api_key <your_openai_api_key>`"
            )
            return

        async with ctx.typing():
            try:
                # Retrieve the image data
                image_bytes, mime_type, source_url = await self.get_image_from_context(
                    ctx, url
                )

                # Query OpenAI Moderation API and track execution time
                start_time = asyncio.get_event_loop().time()
                api_response = await self.query_openai_moderation(
                    api_key, image_bytes, mime_type
                )
                elapsed_time = asyncio.get_event_loop().time() - start_time

                results = api_response.get("results")
                if not results:
                    raise commands.BadArgument(
                        "The OpenAI Moderation API returned an empty or invalid response."
                    )

                result = results[0]
                flagged = result.get("flagged", False)
                categories = result.get("categories", {})
                category_scores = result.get("category_scores", {})

                # Build the beautiful embed
                if flagged:
                    color = discord.Color.from_rgb(239, 68, 68)  # Red/Rose
                    verdict = "⚠️ **Verdict: FLAGGED**"
                    description = "This image contains content that is harmful."
                else:
                    color = discord.Color.from_rgb(34, 197, 94)  # Green/Mint
                    verdict = "✅ **Verdict: SAFE**"
                    description = "No harmful content detected."

                embed = discord.Embed(
                    title="🔍 OpenAI Image Moderation Analysis",
                    description=f"{verdict}\n{description}",
                    color=color,
                )

                # Format the scores list
                category_lines = []
                for cat, is_flagged in categories.items():
                    score = category_scores.get(cat, 0.0)
                    icon = "🚨" if is_flagged else "⚪"
                    formatted_name = cat.replace("/", " / ").replace("_", " ").title()

                    # Highlight flagged ones with bold formatting
                    if is_flagged:
                        line = f"{icon} **{formatted_name}**: **{score * 100:.2f}%**"
                    else:
                        line = f"{icon} {formatted_name}: {score * 100:.2f}%"
                    category_lines.append(line)

                categories_value = "\n".join(category_lines)
                embed.add_field(
                    name="Harm Categories & Scores",
                    value=categories_value or "No data",
                    inline=False,
                )

                # Add metadata fields
                embed.add_field(
                    name="Source URL",
                    value=f"[Link to Image]({source_url})" if len(source_url) < 100 else f"[Link to Image]({source_url[:90]}...)",
                    inline=True,
                )
                embed.add_field(
                    name="Model Used",
                    value="`omni-moderation-latest`",
                    inline=True,
                )

                # Add footer
                embed.set_footer(
                    text=f"API Response: {elapsed_time:.2f}s | Request by {ctx.author}",
                    icon_url=ctx.author.display_avatar.url,
                )

                # Handle image preview based on safety
                if flagged:
                    # Don't directly embed the flagged image.
                    # Provide it in a spoiler link to assist moderators safely.
                    embed.add_field(
                        name="Image Preview",
                        value=f"||[View Flagged Image]({source_url})|| (Hidden for safety)",
                        inline=False,
                    )
                else:
                    # For safe images, display them as thumbnail in the embed
                    embed.set_thumbnail(url=source_url)

                await ctx.send(embed=embed)

            except commands.BadArgument as e:
                # Format friendly error embeds
                error_embed = discord.Embed(
                    title="❌ Moderation Failed",
                    description=str(e),
                    color=discord.Color.from_rgb(239, 68, 68),
                )
                await ctx.send(embed=error_embed)
            except Exception as e:
                log.exception("Unexpected error during image moderation command", exc_info=e)
                error_embed = discord.Embed(
                    title="❌ Unexpected Error",
                    description="An unexpected error occurred. Please contact the bot administrator or check logs.",
                    color=discord.Color.from_rgb(239, 68, 68),
                )
                await ctx.send(embed=error_embed)
