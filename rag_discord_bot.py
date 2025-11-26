import discord
from discord.ext import commands
from discord import app_commands
import configparser
import asyncio
import os
import re
from rag_api_client import query_rag_api

# -----------------------------
# Discord & Client Settings
# -----------------------------
DISCORD_MAX_CHARS = 2000 # The character limit for a single Discord message
TRUNCATION_SUFFIX = "\n\n... (Response cut short to maintain coherence and character limit)"


# --- Configuration Loading ---
def load_config():
    """Loads configuration from config.ini, creating a template if it doesn't exist."""
    config = configparser.ConfigParser()
    config_path = 'config.ini'

    if not os.path.exists(config_path):
        print("config.ini not found. Creating a template now.")
        
        config['DISCORD'] = {'BOT_TOKEN': 'YOUR_DISCORD_BOT_TOKEN_HERE', 'GUILD_ID': '', 'USE_SPECIFIC_CHANNEL': 'False', 'CHANNEL_ID': ''}
        config['RAG_SERVER'] = {'BASE_URL': 'http://0.0.0.0:8000', 'QUERY_ENDPOINT': '/query', 'WEBSEARCH_ENDPOINT': '/websearch_query'}
        
        with open(config_path, 'w') as f:
            config.write(f)
        
        print("Please edit 'config.ini' with your Discord Bot Token and run the script again.")
        exit()

    config.read(config_path)
    
    # Check for required token
    token = config.get('DISCORD', 'BOT_TOKEN')
    if token == 'YOUR_DISCORD_BOT_TOKEN_HERE' or not token:
        print("Error: Please update the BOT_TOKEN in config.ini before running.")
        exit()

    return config

# Load config and set up bot
config = load_config()
BOT_TOKEN = config.get('DISCORD', 'BOT_TOKEN')
GUILD_ID = config.get('DISCORD', 'GUILD_ID') # Optional

RAG_URL_QUERY = config.get('RAG_SERVER', 'BASE_URL') + config.get('RAG_SERVER', 'QUERY_ENDPOINT')
RAG_URL_WEB_QUERY = config.get('RAG_SERVER', 'BASE_URL') + config.get('RAG_SERVER', 'WEBSEARCH_ENDPOINT')

# --- Bot Setup ---
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

# --- Discord Events ---

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    
    if GUILD_ID:
        try:
            guild = discord.Object(id=int(GUILD_ID))
            bot.tree.copy_global_to(guild=guild)
            await bot.tree.sync(guild=guild)
            print(f"Commands synced to Guild ID: {GUILD_ID}")
        except Exception as e:
            print(f"Could not sync commands to guild: {e}. Check GUILD_ID in config.ini.")
            
    else:
        await bot.tree.sync()
        print("Commands synced globally. May take up to 1 hour to appear.")
        
    print('-' * 20)

# --- Slash Command ---

@bot.tree.command(name="ask_local", description="Ask dirty ragger your desire.")
@app_commands.describe(prompt="The question you want to ask.")
async def rag_query_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()

    text_generator = await query_rag_api(RAG_URL, prompt)

    # Calculate the max length for the streaming content *after* the header
    initial_message = (
        f"**Question:** `{prompt}`\n"
        f"**Answer:**\n"
    )

    # 4. Final Update
    final_response_text = text_generator
    final_content = initial_message + final_response_text
    
    await interaction.edit_original_response(content=final_content)

@bot.tree.command(name="ask_web", description="Ask dirty ragger your desire.. But with web search")
@app_commands.describe(prompt="The question you want to ask.")
async def rag_query_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()

    text_generator = await query_rag_api(RAG_URL, prompt)

    # Calculate the max length for the streaming content *after* the header
    initial_message = (
        f"**Question:** `{prompt}`\n"
        f"**Answer:**\n"
    )

    # 4. Final Update
    final_response_text = text_generator
    final_content = initial_message + final_response_text
    
    await interaction.edit_original_response(content=final_content)

if __name__ == '__main__':
    print(f"RAG Server URL: {RAG_URL_QUERY} - {RAG_URL_WEB_QUERY}")
    print("Starting Discord Bot...")
    bot.run(BOT_TOKEN)