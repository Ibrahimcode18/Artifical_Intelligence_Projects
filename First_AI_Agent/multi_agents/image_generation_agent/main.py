# main.py

import asyncio

from workflow_and_agent import run_image_generation_workflow

async def start_chat():
    print("Welcome to the Image Generation Portal!")
    print("How many images would you like to generate?")
    
    while True:
        user_input = input("\n Type here (or type 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            break
            
        # Call the agent workflow
        await run_image_generation_workflow(query=user_input)

if __name__ == "__main__":
    asyncio.run(start_chat())