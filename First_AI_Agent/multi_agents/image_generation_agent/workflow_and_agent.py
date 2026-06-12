# This agent implements the HITL (Human in the loop) where small orders are auto approved and large orders (>5) are paused for human approval.

import asyncio
import uuid
from dotenv import load_dotenv
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.tool_context import ToolContext
from google.adk.runners import Runner
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
from google.adk.sessions import InMemorySessionService

load_dotenv()
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

LARGE_THRESHOLD = 1

def generate_image(
    num_images: int, tool_context: ToolContext
    ) -> dict:
        """Generates images. Requires approval if ordering more than 1 image (LARGE_THRESHOLD).

        Args:
            num_images: Number of images to generate

        Returns:
            Dictionary with order status
        """

        # SCENARIO 1: Small images auto-approve
        if num_images <= LARGE_THRESHOLD:
            return {
                "status": "approved",
                "order_id": f"ORD-{num_images}-AUTO",
                "num_images": num_images,
                "message": f"Order auto-approved: {num_images} images generated",
            }

        # SCENARIO 2: This is the first time this tool is called. Large orders need human approval - PAUSE here.
        if not tool_context.tool_confirmation:  # Check if there is no response yet for approval from the human (i.e. first call)
            tool_context.request_confirmation(  # Now request a response from the human by sending a message back to the Agent and PAUSING execution until human responds
                hint=f"⚠️ Large order: {num_images} images to generate. Do you want to approve?",
                payload={"num_images": num_images},
            )
            return {  # This is sent to the Agent
                "status": "pending",
                "message": f"Order for {num_images} images requires approval",
            }

        # SCENARIO 3: The tool is called AGAIN and is now resuming. Handle approval response - RESUME here.
        if tool_context.tool_confirmation.confirmed: # If human approved the order
            return {
                "status": "approved",
                "order_id": f"ORD-{num_images}-HUMAN",
                "num_images": num_images,
                "message": f"Order approved: {num_images} images generated",
            }
        else:
            return {
                "status": "rejected",
                "message": f"Order rejected: {num_images} images will not be generated",
            }

# Create image generation agent with pausable tool
image_generation_agent = LlmAgent(
    name="image_generation_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are an image generation assistant.
    
When users request to generate images:
1. Use the generate_image tool.
2. If the order status is 'pending', inform the user that approval is required.
3. After receiving the final result, you MUST format your exact response using this strict template:

Order Status: [status]
Order ID: [order_id or N/A]
Number of Images: [number]
""",
    tools=[FunctionTool(func=generate_image)],
)

# Wrap the agent in a resumable app - THIS IS THE KEY FOR LONG-RUNNING OPERATIONS!
image_generation_app = App(
    name="image_generation_coordinator",
    root_agent=image_generation_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
print("✅ Resumable app created!")

session_service = InMemorySessionService()
# Create runner with the resumable app
image_generation_runner = Runner(
    app=image_generation_app,  # Pass the app instead of the agent
    session_service=session_service,
)
print("✅ Runner created!")

def check_for_approval(events):
    """Check if events contain an approval request.

    Returns:
        dict with approval details or None
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None

def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

def create_approval_response(approval_info, approved):
    """Create approval response message."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )

print("✅ Helper functions defined")                    

async def run_image_generation_workflow(query: str):
    """Runs an image generation workflow with approval handling.

    Args:
        query: User's image generation request
    """

    print(f"\n{'='*60}")
    print(f"User > {query}\n")

    # Generate unique session ID
    session_id = f"order_{uuid.uuid4().hex[:8]}"

    # Create session
    await session_service.create_session(
        app_name="image_generation_coordinator", user_id="test_user", session_id=session_id
    )

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    # STEP 1: Send initial request to the Agent. If num_images > 1, the Agent returns the special `adk_request_confirmation` event
    async for event in image_generation_runner.run_async(user_id="test_user", session_id=session_id, new_message=query_content):
        events.append(event)

    # STEP 2: Loop through all the events generated and check if `adk_request_confirmation` is present.
    approval_info = check_for_approval(events)

    # STEP 3: If the event is present, it's a large order - HANDLE APPROVAL WORKFLOW

    if approval_info:
        print(f"⏸️  Pausing for approval...")
            
        user_decision = input(f"\n⚠️  Type 'Y' to approve or 'N' to reject the order: ")
            
        is_approved = user_decision.strip().lower() in ['y', 'yes']
       

        # PATH A: Resume the agent by calling run_async() again with the approval decision
        async for event in image_generation_runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=create_approval_response(
                approval_info, is_approved
            ),  # Send human decision here
            invocation_id=approval_info["invocation_id"],  # Critical: same invocation_id tells ADK to RESUME
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}")

    else:
        # PATH B: If the `adk_request_confirmation` is not present - no approval needed - order completed immediately.
        print_agent_response(events)

    print(f"{'='*60}\n")

print("✅ Workflow function ready")