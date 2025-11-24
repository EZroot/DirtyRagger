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
        
        config['DISCORD'] = {'BOT_TOKEN': 'YOUR_DISCORD_BOT_TOKEN_HERE', 'GUILD_ID': ''}
        config['RAG_SERVER'] = {'BASE_URL': 'http://0.0.0.0:8000', 'QUERY_ENDPOINT': '/query'}
        
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

RAG_URL = config.get('RAG_SERVER', 'BASE_URL') + config.get('RAG_SERVER', 'QUERY_ENDPOINT')

# --- Bot Setup ---
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

# --- Coherent Truncation Function ---

def coherent_truncate(text: str, max_len: int, suffix: str) -> str:
    """
    Truncates a string coherently to a maximum length by looking for the last
    sentence-ending punctuation mark.
    """
    if len(text) <= max_len:
        return text

    # 1. Hard-cut to just before the limit, leaving space for the suffix
    limit = max_len - len(suffix)
    truncated_segment = text[:limit]

    # 2. Find the last sentence-ending punctuation followed by whitespace
    matches = list(re.finditer(r'[.!?](?:\s+|\n\n|\n)', truncated_segment))

    if matches:
        # Get the end index of the last match (the clean cut-off point)
        clean_cut_index = matches[-1].end()
        return truncated_segment[:clean_cut_index] + suffix
    
    # 3. Fallback: If no punctuation is found (e.g., code block), hard truncate
    return truncated_segment.strip() + suffix

# --- Discord Events ---

@bot.event
async def on_ready():
    """Handles logic once the bot is connected and ready."""
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

@bot.tree.command(name="queryrag", description="Query the local RAG server and stream the response.")
@app_commands.describe(prompt="The question you want to ask the RAG system.")
async def rag_query_command(interaction: discord.Interaction, prompt: str):
    """
    The main RAG query command. Handles deferred response and streaming updates.
    """
    await interaction.response.defer()

    confidence, text_generator = await query_rag_api(RAG_URL, prompt)

    full_response = ""
    
    # Calculate the max length for the streaming content *after* the header
    initial_message = (
        f"**RAG Query:** `{prompt}`\n"
        f"**Confidence:** `{confidence:.2f}`\n"
        f"**Answer:**\n"
    )
    # Use the global constant here:
    max_text_len = DISCORD_MAX_CHARS - len(initial_message) - len(TRUNCATION_SUFFIX)
    
    # --- Streaming and Response Updates ---
    last_update_time = asyncio.get_event_loop().time()
    MIN_UPDATE_INTERVAL = 1.0 
    is_truncated = False
    
    # Send the initial header
    await interaction.edit_original_response(content=initial_message)

    try:
        async for chunk in text_generator:
            if is_truncated:
                break

            full_response += chunk
            
            if len(full_response) > max_text_len:
                is_truncated = True

            current_time = asyncio.get_event_loop().time()
            if current_time - last_update_time >= MIN_UPDATE_INTERVAL or is_truncated:
                
                display_text = full_response
                if is_truncated:
                    display_text = coherent_truncate(full_response, max_text_len, TRUNCATION_SUFFIX)
                
                content = initial_message + display_text
                
                await interaction.edit_original_response(content=content)
                last_update_time = current_time

                if is_truncated:
                    break

    except Exception as e:
        error_message = f"An error occurred during streaming: {str(e)}"
        final_content = initial_message + coherent_truncate(full_response, max_text_len, "")
        final_content += f"\n\n**{error_message}**"
        await interaction.edit_original_response(content=final_content[:DISCORD_MAX_CHARS])
        return

    # 4. Final Update
    final_response_text = full_response
    if len(full_response) > max_text_len:
        final_response_text = coherent_truncate(full_response, max_text_len, TRUNCATION_SUFFIX)

    final_content = initial_message + final_response_text
    
    await interaction.edit_original_response(content=final_content)


if __name__ == '__main__':
    print(f"RAG Server URL: {RAG_URL}")
    print("Starting Discord Bot...")
    bot.run(BOT_TOKEN)