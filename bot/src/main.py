# local modules
#import ocr
import os
from dotenv import load_dotenv
# source modules
import bot
import test

# load environment
load_dotenv()
ENV: str = os.getenv("ENVIRONMENT")

# main entry point
def main() -> None:
    if ENV == "prod":
        print("Loading Discord Bot.")
        bot.main()
    elif ENV == "test":
        print("Loading Test Suite")
        #ocr.main()
        test.main()
    else:
        print("Error with ENVIRONMENT in .env file. Must be 'test' or 'prod'.")

if __name__ == "__main__":
    main()
