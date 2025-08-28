# Persona-Based News Recommender

A comprehensive news recommendation system that generates detailed user personas from click history and uses them to provide personalized recommendations with strong disengagement prevention.

## Quick Start

```bash
# Core functionality test
python3 test_core.py

# Integration test (needs full POPROX environment)  
python3 tests/test_persona_recommender.py

# Full standalone demo with LLM
python3 tests/persona_demos/demo_persona_standalone.py

# Multi-user comparison demo
python3 tests/persona_demos/persona_comparison_demo.py
```

## Key Components

- `src/poprox_recommender/components/embedders/user_persona.py` - LLM-powered persona generation with disengagement analysis
- `src/poprox_recommender/components/scorers/persona_scorer.py` - Persona-based article scoring with engagement patterns
- `src/poprox_recommender/recommenders/configurations/user_persona_recommender.py` - POPROX-compatible pipeline configuration

## Environment Setup

Create `.env` file in project root:
```
GEMINI_API_KEY="your_api_key_here"
```

## Integration with POPROX

The persona recommender integrates seamlessly with existing POPROX framework:

- Uses same `configure(builder, num_slots, device)` pattern as `nrms_topic_scores`
- Compatible with existing NRMS embedders and models
- Can be added to evaluation configs alongside existing recommenders
- Provides graceful fallback when dependencies unavailable

## Key Features

- **LLM-Powered Analysis**: Uses Gemini API for comprehensive persona generation
- **Disengagement Intelligence**: Focuses on what users actively avoid
- **Content Fatigue Prevention**: Prevents recommendation burnout
- **Multi-Method Approach**: LLM → Pretrained → Simple (robust fallbacks)
- **Individual Psychology**: Understanding beyond just topic preferences
- **Production Ready**: Error handling, caching, performance optimization

## Performance

- **With API key**: Full LLM persona analysis (comprehensive but slower)
- **Without API key**: Simple embedding aggregation (fast and effective)
- **Fallback chain**: Ensures system always works regardless of dependencies