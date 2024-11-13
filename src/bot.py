import os
import random
import datetime
import time
from dotenv import load_dotenv
from discord import Intents, Client, Message, Emoji, File, Attachment
from discord import app_commands
from discord.ext import commands
from typing import Final
# src modules
import responses

# load token safely
load_dotenv()
TOKEN: Final[str] = os.getenv("DISCORD_TOKEN")
LOG_ID: int = os.getenv("LOG_ID")

# setup bot
intents: Intents = Intents.default()
intents.reactions = True
bot = commands.Bot(command_prefix="unused_", intents=intents)

# load templates into memory when bot is launched
# should be in a config file
template_filename="images/template_large.png"
list_of_templates = responses.load_templates(template_filename)
assert list_of_templates is not None, "file could not be read, check with os.path.exists()"

# load this once
log_channel = None

# process image
async def process_image(interaction, attachment) -> str:
    try:
        image_data = await attachment.read()
        try:
            response: [str] = await responses.get_response_from_ocr(interaction.id, image_data, list_of_templates)
            return response
        except Exception as e:
            print(e)
            return "Error processing image."
    except Exception as e:
        print(e)
        return "Error reading image."

# handle startup of bot
@bot.event
async def on_ready() -> None:
    print(f"{bot.user} is now running.")
    try:
        global log_channel
        print(f"Started at: {datetime.datetime.now()}")
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} commands(s)")
        print("Available commands:", [command.name for command in bot.tree.get_commands()])
        log_channel = bot.get_channel(int(LOG_ID))
        if log_channel is None:
            print(f"WARNING: Could not connect to log channel with ID: {LOG_ID}")
        else:
            print(f"Succesfully connected to log channel: {log_channel.name}")
    except Exception as e:
        print(f"Error syncing commands: {e}")


@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    # Optional: inform users about slash commands if they try to use prefix commands
    if message.content.startswith('!'):
        print(f"User attempted prefix command: {message.content}")
        await message.channel.send("Please use slash commands: `/ping` or `/ocr`")


@bot.event
async def on_application_command_error(interaction, error):
    print(f"Application command error: {str(error)}")
    await interaction.followup.send("An error occured while processing the command", ephemeral=True);


# slash command to run ocr
@bot.tree.command(name="ocr", description="Process a replaycode image with OCR.")
async def ocr(interaction, image: Attachment):
    print(f"OCR command received at: {datetime.datetime.now()} from {interaction.user} in {interaction.guild}")
    await interaction.response.defer()

    media_type = image.content_type.lower().split("/")
    if media_type[0] != "image" or media_type[1] == "gif":
        print(f"Invalid image type {media_type}")
        await interaction.followup.send("Please provide a valid image (not a GIF)")
        return

    try:
        response = await process_image(interaction, image)
        print(f"OCR Processing complete at: {datetime.datetime.now()} for {interaction.user} in {interaction.guild}")
        #message = await interaction.followup.send(image, response)
        message = await interaction.followup.send(content=response, file=await image.to_file())
        # with this reactions won't be added in dms
        if interaction.guild:
            await message.add_reaction("\u2705")
            await message.add_reaction("\u274c")
    except Exception as e:
        print(f"Error: {e}")
        await interaction.followup.send("An error occured while processing the image.")
        
    print(f"[{interaction.guild} - {interaction.channel}] {interaction.user}: {image.url}")
    #print(response)

@bot.tree.command(name="help", description="Help!.")
async def help(interaction):
    print(f"Help command received from {interaction.user}")
    await interaction.response.send_message(f"Go here: https://github.com/afnleaf/ow2-replaycode-ocr")


@bot.tree.command(name="ping", description="Check if the bot is responsive")
async def ping(interaction):
    print(f"Ping command received from {interaction.user}")
    await interaction.response.send_message(f"Pong! Latency: {round(bot.latency * 1000)}ms")


# handle incoming reactions
@bot.event
async def on_raw_reaction_add(payload):
    channel = bot.get_channel(payload.channel_id)
    # ignore when in dm or bot adding two reactions on message creation
    if not payload.member or payload.member == bot.user:
        return

    emoji = payload.emoji.name

    #print(payload.message_id)
    if emoji == "✅":
        print(".nice.")
        #await process_message_id(payload.channel_id, payload.message_id, True)
        msg = await channel.fetch_message(payload.message_id)
        n = random.random() * 4
        print(n)
        
        if n >= 0 and n < 1:
            content = "🍦"
        elif n >= 1 and n < 2:
            content = "🍨"
        elif n >= 2 and n < 3:
            content = "🍧"
        elif n >= 3 and n <= 4:
            content = "🍕"

        if n < 2:            
            await msg.add_reaction(content)

    elif emoji == "❌":
        print(".uhoh.")
        #await process_message_id(payload.channel_id, payload.message_id, False)
        await log_bad_message(await channel.fetch_message(payload.message_id), payload.member)

    elif emoji == "🍕":
        msg = await channel.send("mama mia")
        time.sleep(2)
        await msg.delete()

    elif emoji == "🍧":
        msg = await channel.send("👅👅👅")
        time.sleep(2)
        await msg.delete()

    elif emoji == "🍦" or emoji == "🍨":
        msg = await channel.send("mmm ice cream so good, yes yes yes, gang gang, gang gang")
        time.sleep(2)
        await msg.delete()


async def log_bad_message(message, member):
    try:
        attachments = [attachment.url for attachment in message.attachments]
        log_entry = f"""
**Time of log:** {datetime.datetime.now()}
**Time of message:** {message.created_at}
**Message ID:** {message.id}
**Author:** {message.author}
**Channel:** {message.channel}
**Guild:** {message.guild}
**Attachments:** {', '.join(attachments) if attachments else 'None'}
**Reacted by:** {member}
**Content:**\n {message.content}
                """
        if log_channel:
            await log_channel.send(log_entry)
        else:
            print(f"Log Channel {LOG_ID} not found.")
    except Exception as e:
        print(f"Error while logging message from reaction: {e}")


# reaction adding process prints image url in logs
# collect .uhoh. to .end. to further improve detection algorithm
async def process_message_id(channel_id, message_id, log_status):
    channel = bot.get_channel(channel_id)
    if not channel:
        print(f"Channel {channel_id} not found.")
        return

    # logic to get image
    print(message_id)
    response_message = await channel.fetch_message(message_id)
    print(response_message.content)
    parts = response_message.content.split("\n")
    image_message_id = parts[0]
    image_message = await channel.fetch_message(image_message_id)


    #print(f"img_id: {image_message_id}")
    if image_message.attachments:
        for attachement in image_message.attachments:
            print(attachement.url)
            # post to log channel
            content = f"[{image_message.guild} - {image_message.channel}] {image_message.author} - "
            if log_status:
                content += f"✅nice: {attachement.url}"
                meow = await image_message.channel.send("meow")
                time.sleep(2)
                await meow.delete()
            else:
                content += f"❌uhoh: {attachement.url}"
            log_channel = bot.get_channel(int(LOG_ID))
            await log_channel.send(content)
    print(".end.")


# main entry point
def main() -> None:
    bot.run(token=TOKEN)


if __name__ == "__main__":
    main()

