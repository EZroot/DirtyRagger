import discord
from discord.ext import commands
from discord import app_commands
import configparser
import asyncio
import os
from rag_api_client import query_rag_api # Import the client function

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
# Setup the bot intents. Since we are only using slash commands, 
# we can safely use Intents.default() without requiring the privileged 'message_content' intent.
intents = discord.Intents.default()
# intents.message_content = True # REMOVED: No longer necessary, fixing the PrivilegedIntentsRequired error

# Initialize the bot with intents
bot = commands.Bot(command_prefix='!', intents=intents)

# --- Discord Events ---

@bot.event
async def on_ready():
    """Handles logic once the bot is connected and ready."""
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    
    if GUILD_ID:
        # Register commands to a specific guild for faster testing
        try:
            guild = discord.Object(id=int(GUILD_ID))
            bot.tree.copy_global_to(guild=guild)
            await bot.tree.sync(guild=guild)
            print(f"Commands synced to Guild ID: {GUILD_ID}")
        except Exception as e:
            print(f"Could not sync commands to guild: {e}. Check GUILD_ID in config.ini.")
            
    else:
        # Register global commands (takes up to 1 hour)
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
    # 1. Defer the response immediately. This tells Discord we need time to process.
    await interaction.response.defer()

    # 2. Call the asynchronous client function
    confidence, text_generator = await query_rag_api(RAG_URL, prompt)

    # 3. Handle Streaming and Response Updates
    full_response = ""
    # Start with a message header that includes the confidence score
    initial_message = (
        f"**RAG Query:** `{prompt}`\n"
        f"**Confidence:** `{confidence:.2f}`\n"
        f"**Answer:**\n"
    )
    
    # Use a variable to track the last time we updated Discord, to avoid hitting rate limits
    last_update_time = asyncio.get_event_loop().time()
    MIN_UPDATE_INTERVAL = 1.0 # Update Discord max once per second
    
    # Send the initial header
    await interaction.edit_original_response(content=initial_message)


    try:
        # Iterate over the text chunks from the generator
        async for chunk in text_generator:
            full_response += chunk
            
            current_time = asyncio.get_event_loop().time()
            # Only update the Discord message if enough time has passed
            if current_time - last_update_time >= MIN_UPDATE_INTERVAL:
                # Truncate response to Discord's 2000 character limit, saving space for header
                content = initial_message + full_response
                if len(content) > 2000:
                    content = initial_message + full_response[:1900] + "\n... (truncated)"
                
                await interaction.edit_original_response(content=content)
                last_update_time = current_time

    except Exception as e:
        error_message = f"An error occurred during streaming: {str(e)}"
        await interaction.edit_original_response(content=initial_message + full_response + f"\n\n**{error_message}**")
        return

    # 4. Final Update: Ensure the complete, final text is sent
    final_content = initial_message + full_response
    if len(final_content) > 2000:
        final_content = initial_message + full_response[:1990] + "\n\n...(Full text exceeded 2000 characters)"

    await interaction.edit_original_response(content=final_content)


if __name__ == '__main__':
    print(f"RAG Server URL: {RAG_URL}")
    print("Starting Discord Bot...")
    bot.run(BOT_TOKEN)