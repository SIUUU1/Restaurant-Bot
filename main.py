## Restaurant Bot

import asyncio

from dotenv import load_dotenv

# Load environment variables (e.g. OPENAI_API_KEY) from a local .env file.
load_dotenv()

from agents import Agent, Runner, RunContextWrapper, function_tool, handoff
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


# ===========================================================================
# 1. Mock data (use a DB / external API in a real service)
# ===========================================================================
MENU = {
    "마르게리타 피자": {"price": 15000, "allergens": ["밀", "유제품"], "veg": True},
    "페퍼로니 피자": {"price": 18000, "allergens": ["밀", "유제품", "돼지고기"], "veg": False},
    "버섯 리조또": {"price": 16000, "allergens": ["유제품"], "veg": True},
    "시저 샐러드": {"price": 12000, "allergens": ["유제품", "계란", "생선"], "veg": False},
    "가든 샐러드": {"price": 10000, "allergens": [], "veg": True},
    "티라미수": {"price": 8000, "allergens": ["밀", "유제품", "계란"], "veg": True},
}

# Simple in-memory stores for orders and reservations.
ORDERS: list[dict] = []
RESERVATIONS: list[dict] = []


# ===========================================================================
# 2. Tools (function tools) used by each specialist
#    - The type hints define the input schema; the docstring becomes the
#      tool description.
# ===========================================================================
@function_tool
def get_menu(vegetarian_only: bool = False) -> str:
    """Return the full menu. If vegetarian_only=True, return only vegetarian items."""
    lines = []
    for name, info in MENU.items():
        if vegetarian_only and not info["veg"]:
            continue
        veg_tag = " (vegetarian 🌱)" if info["veg"] else ""
        lines.append(f"- {name}{veg_tag}: {info['price']:,}원")
    return "\n".join(lines) if lines else "No items match the requested criteria."


@function_tool
def check_allergens(dish_name: str) -> str:
    """Return the allergens contained in a specific dish."""
    info = MENU.get(dish_name)
    if not info:
        return f"Could not find a dish named '{dish_name}'. Please check the name."
    allergens = ", ".join(info["allergens"]) if info["allergens"] else "none"
    return f"Allergens in {dish_name}: {allergens}"


@function_tool
def place_order(items: list[str]) -> str:
    """Take a list of dish names, record the order, and compute the total."""
    confirmed, total = [], 0
    for item in items:
        info = MENU.get(item)
        if info:
            confirmed.append(item)
            total += info["price"]
    if not confirmed:
        return "None of those items are orderable. Please check the dish names."
    order_id = len(ORDERS) + 1
    ORDERS.append({"id": order_id, "items": confirmed, "total": total})
    return f"Order #{order_id} confirmed — {', '.join(confirmed)} / total {total:,}원"


@function_tool
def make_reservation(name: str, party_size: int, date_time: str) -> str:
    """Create a table reservation from the name, party size, and desired date/time."""
    res_id = len(RESERVATIONS) + 1
    RESERVATIONS.append(
        {"id": res_id, "name": name, "party_size": party_size, "date_time": date_time}
    )
    return (
        f"Reservation #{res_id} confirmed — {name}, {party_size} guests, {date_time}. "
        "We look forward to your visit!"
    )


# ===========================================================================
# 3. Callback that surfaces a handoff in the UI
#    on_handoff fires the MOMENT the LLM calls a transfer_to_* tool.
#    The target's display name is bound via a closure so each handoff prints
#    its own message.
# ===========================================================================
def make_handoff_logger(display_name: str):
    async def _on_handoff(ctx: RunContextWrapper) -> None:
        print(f"\n   🔄 [Connecting you to {display_name}...]\n")

    return _on_handoff


# ===========================================================================
# 4. Specialist agents
#    - handoff_description: hint that tells Triage when to pick this specialist
#    - prompt_with_handoff_instructions: SDK-recommended handoff prompt + role
# ===========================================================================
menu_agent = Agent(
    name="Menu Agent",
    handoff_description="Specialist for food information: menu items, ingredients, allergens, and vegetarian options",
    instructions=prompt_with_handoff_instructions(
        "You are the restaurant's Menu specialist. "
        "The moment you take over a conversation, read the customer's MOST RECENT message and act on it "
        "in this same turn by calling the right tool immediately. Never ask the customer to repeat themselves "
        "and never return an empty reply.\n"
        "- Vegetarian menu request -> call get_menu(vegetarian_only=True) and list the results\n"
        "- Full menu request -> call get_menu() and list the results\n"
        "- Allergen question about a specific dish -> call check_allergens\n"
        "Answer concretely based on the tool result, and always reply in friendly Korean. "
        "Hand off order requests to the Order Agent and reservation requests to the Reservation Agent."
    ),
    tools=[get_menu, check_allergens],
)

order_agent = Agent(
    name="Order Agent",
    handoff_description="Specialist for taking and confirming food orders",
    instructions=prompt_with_handoff_instructions(
        "You are the Order specialist. Once you have confirmed which dishes the customer wants, "
        "immediately call the place_order tool, then tell them the order number and total amount. "
        "Hand off detailed menu questions to the Menu Agent and reservation requests to the Reservation Agent. "
        "Always reply in friendly Korean."
    ),
    tools=[place_order],
)

reservation_agent = Agent(
    name="Reservation Agent",
    handoff_description="Specialist for handling table reservations",
    instructions=prompt_with_handoff_instructions(
        "You are the Reservation specialist. Ask for the information needed to book a table "
        "(reservation name, party size, desired date/time) one item at a time. "
        "Once you have all of it, call the make_reservation tool to create the booking and then share the reservation number. "
        "Hand off menu questions to the Menu Agent and order requests to the Order Agent. Always reply in friendly Korean."
    ),
    tools=[make_reservation],
)


# ===========================================================================
# 5. Triage (routing) agent
# ===========================================================================
triage_agent = Agent(
    name="Triage Agent",
    handoff_description="Front desk that analyzes customer requests and routes them to the right specialist",
    instructions=prompt_with_handoff_instructions(
        "You are the restaurant's front desk. Analyze the customer's request and IMMEDIATELY hand off (transfer) "
        "to the most appropriate specialist.\n"
        "- Menu / ingredient / allergen / vegetarian questions -> Menu Agent\n"
        "- Food orders -> Order Agent\n"
        "- Table reservations -> Reservation Agent\n"
        "Never answer directly or stop after a greeting. Your only job is to hand off. "
        "When the right specialist is clear, call that handoff tool right away without any extra explanation."
    ),
)


# ===========================================================================
# 6. Handoff wiring (mesh)
#    - Triage is the initial routing hub.
#    - Every specialist can hand off DIRECTLY to any other specialist, so a
#      topic change is resolved in a single handoff (e.g. while making a
#      reservation, a menu question goes straight to the Menu Agent).
# ===========================================================================
DISPLAY_NAME = {
    "Triage Agent": "the front desk",
    "Menu Agent": "the menu specialist",
    "Order Agent": "the order specialist",
    "Reservation Agent": "the reservation specialist",
}


def link(target: Agent) -> handoff:
    """Build a handoff object to the target agent and attach the display callback."""
    return handoff(target, on_handoff=make_handoff_logger(DISPLAY_NAME[target.name]))


triage_agent.handoffs = [link(menu_agent), link(order_agent), link(reservation_agent)]
menu_agent.handoffs = [link(triage_agent), link(order_agent), link(reservation_agent)]
order_agent.handoffs = [link(triage_agent), link(menu_agent), link(reservation_agent)]
reservation_agent.handoffs = [link(triage_agent), link(menu_agent), link(order_agent)]


# ===========================================================================
# 7. Chat loop
#    - Keeps the conversation history so context (e.g. "before that...") carries over.
#    - Continues from whichever agent answered last (last_agent); the first
#      turn starts from the Triage Agent.
# ===========================================================================
async def main() -> None:
    print("🍽️  Restaurant bot here. How can I help?  (type 'quit' to exit)\n")

    history: list[dict] = []        # accumulated conversation input
    current_agent = triage_agent    # the agent currently in charge of replying

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThanks for stopping by!")
            break

        if user_input.lower() in {"quit", "exit", "종료"}:
            print("Thank you. Have a great day!")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        result = await Runner.run(current_agent, history)

        print(f"{result.last_agent.name}: {result.final_output}\n")

        # Update history and current agent for the next turn.
        history = result.to_input_list()
        current_agent = result.last_agent


if __name__ == "__main__":
    asyncio.run(main())