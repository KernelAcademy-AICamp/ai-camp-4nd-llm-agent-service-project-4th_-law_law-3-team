"""python -m scripts.news_pipeline 실행 지원"""

import asyncio

from scripts.news_pipeline.cli import main

asyncio.run(main())
