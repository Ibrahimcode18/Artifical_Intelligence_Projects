# main.py

import asyncio

from shipping_agent import run_shipping_workflow

async def start_chat():
    print("Welcome to the Shipping Portal!")
    print("What would you like to ship today?")
    
    while True:
        user_input = input("\n Type here (or type 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            break
            
        # Call the agent workflow
        await run_shipping_workflow(query=user_input)

if __name__ == "__main__":
    asyncio.run(start_chat())