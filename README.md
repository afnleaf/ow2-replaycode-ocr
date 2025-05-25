# Replaycode-ocr Discord Bot

## What
There are two sides of this repo, the discord bot and the tesseract-ocr finetune (work in progress). The finetuned model will be made available when finished.

## Help

## Install
https://discord.com/oauth2/authorize?client_id=1214396853374812211

- Click link to add the bot to your server
- Limit the channels the bot has access to
- DM the bot if you want to
- Bot will only process images it can template match
- Run command `/ocr`

If you would like to run a self hosted version of the bot reach out to me -> @afnckingleaf on twitter or discord.

## Update permissions
https://discord.com/oauth2/authorize?client_id=1214396853374812211&integration_type=0&scope=applications.commands

For the updated version, the bot needs permisionns to be able to run and create commands.

## Commands

`/ocr`

Type this command, then attach an image and press enter. This command will do the image processing for you.

`/ping`

See how fast the connection to the server is.

`/help`

Links you to this page.

## How it works
You input a replay code image such as this one:

![](/bot/images/image_case7.png)

The process works by looking for this template:

![](/bot/images/template_large.png)

![](/bot/images/boxes.png)

Once the template is found, the replaycode text is isolated and then processed.

![](/bot/images/before2.png)

![](/bot/images/after_2.png)

Now finally Tesseract-OCR or the Granite Vision 3.2 can work their magic!

----

## Features
- <s>Post picture in a channel the bot sees, it spits out replay codes as text, but only if the template is matched.</s>
- /ocr command with an attached picture, the bots spits out replay codes as text.
- React to the message if it was right or wrong to help out with improving the character recognition.

## Problems
- Pytesseract character recognition issues
    - X vs 4
    - Q vs 0
    - 0 vs 6

## ToDo:
- [x] message bot directly
- [x] accept images only
- [x] create message then edit it (let user know bot is calculating)
- [x] template match for high pixel density
- [x] variable high density sizes for crop
- [x] logging for testing
- [x] reactions to output
- [x] refactor respond_to_message()
- [x] logs go to server
- [x] sort by order found in image
- [x] testing cases for accuracy increase/decrease with ocr performance testing
- [x] fixing ocr performance
- [x] prod vs test env
- [x] remove priviledged intents
- [ ] slash commands
- [ ] improve OCR

## Environment

```
DISCORD_TOKEN=<token>
LOG_ID=<id>
ENVIRONMENT= test | prod
```
