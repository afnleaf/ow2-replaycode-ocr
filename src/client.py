import os
import random
import time
from discord import Intents, Client, Message, Emoji
from dotenv import load_dotenv
from typing import Final
# local modules
import responses

# load token safely
load_dotenv()
TOKEN: Final[str] = os.getenv("DISCORD_TOKEN")
LOG_ID: int = os.getenv("LOG_ID")

# setup bot
intents: Intents = Intents.default()
intents.message_content = False 
intents.messages = True
intents.reactions = True
#intents.mentions = True
client: Client = Client(intents=intents)

# load templates into memory when bot is launched
# should be in a config file
template_filename="images/template_large.png"
list_of_templates = responses.load_templates(template_filename)
assert list_of_templates is not None, "file could not be read, check with os.path.exists()"


# message functionality
async def respond_to_message(message: Message) -> None:
    try:
        if message.attachments:
            channel = message.channel
            for attachment in message.attachments:
                # accept any image except for gif
                media_type = attachment.content_type.lower().split("/")
                if media_type[0] == "image" and not media_type[1] == "gif" and attachment.url:
                    await process_attachment(message, attachment)
    except Exception as e:
        print(e)


# process actual image found in message
async def process_attachment(message, attachment) -> None:    
    try:
        # load image
        image_data = await attachment.read()
    
        # tell user image is being processed
        message_to_channel = await message.channel.send("Processing input...")

        # process image
        try:
            response: [str] = await responses.get_response_from_ocr(message.id, image_data, list_of_templates)
            await message_to_channel.edit(content=response)
        except Exception as e:
            print(e)
            await message_to_channel.edit(content="Error.")
            time.sleep(1)
            await message_to_channel.delete()

        # reactions
        # with this reactions won't be added in dms
        if message.guild:
            await message_to_channel.add_reaction("✅")
            await message_to_channel.add_reaction("❌")

        # log for testing
        print(f"[{message.guild} - {message.channel}] {message.author}: {attachment.url}")
        print(response)

    except Exception as e:
        print(e)


# reaction adding process prints image url in logs
# collect .uhoh. to .end. to further improve detection algorithm
async def process_message_id(channel_id, message_id, log_status):
    channel = client.get_channel(channel_id)
    print(channel_id)
    if channel:
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
                log_channel = client.get_channel(int(LOG_ID))
                await log_channel.send(content)
        print(".end.")

    return  


# handle startup of bot
@client.event
async def on_ready() -> None:
    print(f"{client.user} is now running.")


# handle incoming messages
@client.event
async def on_message(message: Message) -> None:
    # ignore self
    if message.author == client.user:
        return
    # channel setup to make sure bot only monitors channel with vods?
    # check what channel message is in

    #if not (client.user in message.mentions or isinstance(message.channel, discord.DMChannel)):
    #    return

    if not client.user.mentioned_in(message):
        return

    # respond to message in channels where bot sees
    await respond_to_message(message)


# handle incoming reactions
@client.event
async def on_raw_reaction_add(payload):
    channel = client.get_channel(payload.channel_id)
    # ignore when in dm or bot adding two reactions on message creation
    if not payload.member or payload.member == client.user:
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
        await process_message_id(payload.channel_id, payload.message_id, False)

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

    
# main entry point
def main() -> None:
    client.run(token=TOKEN)


if __name__ == "__main__":
    main()
