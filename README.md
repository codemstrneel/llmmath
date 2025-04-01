Create a .env file in the root directory and set
`OPENAI_API_KEY=<YOUR KEY>`
In your Python file, run
```
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
```
