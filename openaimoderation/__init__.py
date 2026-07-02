from redbot.core.bot import Red

from .main import OpenAIModeration


async def setup(bot: Red) -> None:
    await bot.add_cog(OpenAIModeration(bot))
