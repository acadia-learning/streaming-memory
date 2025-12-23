"""
Sample first-person memories for testing the Hebbian memory system.

Diverse range of:
- Topics (work, relationships, hobbies, travel, emotions, facts)
- Time periods (recent to old)
- Emotional intensity (mundane to significant)
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import MemoryPool


def days_ago(n: int) -> datetime:
    """Helper to create dates relative to now."""
    return datetime.now() - timedelta(days=n)


def hours_ago(n: int) -> datetime:
    """Helper to create dates relative to now."""
    return datetime.now() - timedelta(hours=n)


# Format: (content, emotional_intensity, created_at)
SAMPLE_MEMORIES = [
    # Recent work memories
    ("I had a productive meeting with Sarah about the Q4 roadmap today. We decided to prioritize the mobile app redesign.", 0.4, hours_ago(3)),
    ("The deployment failed at 2am last night. I spent three hours debugging before finding it was a config issue.", 0.7, days_ago(1)),
    ("Got positive feedback on my presentation to the exec team. They seemed impressed with the data visualization.", 0.8, days_ago(2)),
    ("I need to review the pull requests from the junior developers before Friday.", 0.2, hours_ago(6)),
    
    # Relationships
    ("Mom called yesterday. She's planning to visit next month for my birthday.", 0.6, days_ago(1)),
    ("Had dinner with Alex at that new Italian place. The pasta was incredible but service was slow.", 0.5, days_ago(4)),
    ("Jake and I had an argument about the project direction. I think we need to talk it through.", 0.7, days_ago(3)),
    ("My sister announced she's pregnant! I'm going to be an uncle.", 0.95, days_ago(7)),
    ("Caught up with college friends over video call. Marcus is moving to Seattle.", 0.6, days_ago(14)),
    
    # Hobbies and interests
    ("Started reading that sci-fi book Emma recommended. It's about a colony ship traveling to a distant star.", 0.4, days_ago(5)),
    ("Went for a 5k run this morning. My pace is improving - hit 24 minutes.", 0.5, hours_ago(8)),
    ("The photography workshop was great. Learned about composition and the rule of thirds.", 0.6, days_ago(10)),
    ("Finally beat that difficult boss in Elden Ring after 20 attempts.", 0.7, days_ago(2)),
    ("My sourdough starter is finally active. Made my first loaf yesterday.", 0.6, days_ago(3)),
    
    # Travel and experiences
    ("The trip to Japan last year was incredible. Kyoto's temples were so peaceful.", 0.9, days_ago(365)),
    ("Got stuck in traffic for 2 hours on the way to the airport. Almost missed my flight.", 0.6, days_ago(30)),
    ("The hiking trip to Yosemite was breathtaking. Half Dome was harder than expected.", 0.85, days_ago(90)),
    ("Visited the new art museum downtown. The modern art section was confusing but interesting.", 0.4, days_ago(21)),
    
    # Emotional/significant
    ("I've been feeling anxious about the upcoming performance review.", 0.7, days_ago(2)),
    ("Had a great therapy session. We talked about setting boundaries at work.", 0.6, days_ago(7)),
    ("The sunset from my balcony tonight was stunning. Orange and pink streaks across the sky.", 0.5, hours_ago(2)),
    ("I realized I've been neglecting my health. Need to get back to regular exercise.", 0.5, days_ago(4)),
    ("Feeling grateful for my support system. Friends and family have been there for me.", 0.7, days_ago(6)),
    
    # Facts and learnings
    ("Discovered that the coffee shop on 5th Street has the best espresso in town.", 0.4, days_ago(12)),
    ("My car needs an oil change. The dashboard light came on yesterday.", 0.3, days_ago(1)),
    ("Learned that my neighbor is a retired pilot. He has amazing stories.", 0.5, days_ago(8)),
    ("The new Thai restaurant by the office is too spicy for me.", 0.3, days_ago(15)),
    ("Found out my favorite podcast is ending after this season.", 0.6, days_ago(5)),
    
    # Mundane/low intensity
    ("Bought groceries. Need to remember to get more milk next time.", 0.1, days_ago(2)),
    ("The weather has been unusually warm for December.", 0.2, days_ago(1)),
    ("Changed the light bulb in the bathroom.", 0.1, days_ago(4)),
    ("The wifi was slow at the coffee shop today.", 0.2, hours_ago(5)),
    
    # Older memories with high emotional intensity
    ("Graduated from college five years ago. That feeling of accomplishment was incredible.", 0.9, days_ago(1825)),
    ("My first day at my current job was terrifying but exciting.", 0.8, days_ago(730)),
    ("Lost my dog Max two years ago. I still miss him sometimes.", 0.95, days_ago(730)),
    ("The concert where I met my best friend. We bonded over our love of indie music.", 0.85, days_ago(1000)),
    
    # Random facts
    ("I'm allergic to shellfish. Found out the hard way at a wedding.", 0.7, days_ago(500)),
    ("My password pattern is usually something with my birth year and an exclamation mark.", 0.3, days_ago(100)),
    ("I prefer window seats on flights. Something about seeing the clouds.", 0.2, days_ago(60)),
    ("I've been trying to learn Spanish. Duolingo says I'm on a 50 day streak.", 0.4, days_ago(50)),
    ("My favorite color is deep blue, like the ocean at dusk.", 0.3, days_ago(200)),
    
    # Work-specific knowledge
    ("The API rate limit is 1000 requests per minute. We hit it during the Black Friday sale.", 0.5, days_ago(25)),
    ("Our main database is PostgreSQL, but we use Redis for caching.", 0.3, days_ago(100)),
    ("The design system uses Figma. I have edit access to the component library.", 0.2, days_ago(45)),
    
    # Health and wellness
    ("My annual checkup went well. Doctor said my cholesterol is a bit high.", 0.5, days_ago(60)),
    ("Started taking vitamin D supplements. Apparently I was deficient.", 0.3, days_ago(30)),
    ("Tried meditation for the first time. It was harder to quiet my mind than expected.", 0.4, days_ago(14)),
    
    # Future plans
    ("Planning to take a vacation in March. Thinking about Portugal or Spain.", 0.5, days_ago(7)),
    ("Want to learn guitar this year. Already bookmarked some YouTube tutorials.", 0.4, days_ago(10)),
    ("Considering getting a cat. The shelter nearby has some adorable ones.", 0.5, days_ago(3)),
]


def load_sample_memories(pool: "MemoryPool") -> int:
    """Load all sample memories into the pool."""
    for content, emotion, created_at in SAMPLE_MEMORIES:
        pool.add(
            content=content,
            emotional_intensity=emotion,
            created_at=created_at,
        )
    return len(SAMPLE_MEMORIES)


