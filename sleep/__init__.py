from .main import eep

async def setup(bot):
    await bot.add_cog(eep(bot))
