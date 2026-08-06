"""Writes eval_harness/dataset/golden_set.jsonl -- the hand-curated golden dataset for the
demo RAG pipeline (Phase 4). Every ground_truth_contexts entry below is a verbatim excerpt
verified against the actual corpus files in rag_demo/data/corpus/ (not paraphrased), the same
discipline that caught the Phase 3 corpus-splicing bug.

Re-run to regenerate golden_set.jsonl from this source of truth after editing entries here.
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "eval_harness" / "dataset" / "golden_set.jsonl"

QA_PAIRS: list[dict] = [
    # ---------------------------------------------------------------- direct (14)
    {
        "id": "q001",
        "question": "What is the minimal code needed to create a working FastAPI app with one GET endpoint?",
        "ground_truth_answer": (
            "Create a FastAPI() instance and decorate an async function with @app.get(\"/\") "
            "for the route you want, e.g.:\n\n"
            "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get(\"/\")\nasync def root():\n"
            "    return {\"message\": \"Hello World\"}"
        ),
        "ground_truth_contexts": [
            "app = FastAPI()",
            "@app.get(\"/\")\nasync def root():",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q002",
        "question": "If I declare a path parameter as item_id: int, what happens if a client sends a non-integer value in the URL?",
        "ground_truth_answer": (
            "FastAPI validates the value against the declared type automatically. If the client "
            "sends a value that isn't a valid integer, FastAPI returns an HTTP error response "
            "(422 Unprocessable Entity) instead of running the path operation function."
        ),
        "ground_truth_contexts": [
            "You can declare the type of a path parameter in the function, using standard Python type annotations",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q003",
        "question": "How do I make a query parameter optional in FastAPI?",
        "ground_truth_answer": (
            "Give it a default value of None (typically typed as str | None = None). Any "
            "non-path parameter with a default value is treated as optional."
        ),
        "ground_truth_contexts": [
            "When you declare a default value for non-path parameters (for now, we have only seen query parameters), then it is not required.",
            "If you don't want to add a specific value but just make it optional, set the default as `None`.",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q004",
        "question": "How do I enforce a minimum length on a query parameter's string value?",
        "ground_truth_answer": (
            "Use Query() with the min_length argument, e.g. "
            "q: Annotated[str | None, Query(min_length=3)] = None."
        ),
        "ground_truth_contexts": [
            "q: Annotated[str | None, Query(min_length=3)] = None",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q005",
        "question": "What does setting ge=1 on a path parameter's Path() validation mean?",
        "ground_truth_answer": (
            "It means the value must be an integer greater than or equal to 1 -- 'ge' stands "
            "for 'greater than or equal'."
        ),
        "ground_truth_contexts": [
            "Here, with `ge=1`, `item_id` will need to be an integer number \"greater than or equal\" to `1`.",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q006",
        "question": "How do I declare a request body in FastAPI?",
        "ground_truth_answer": (
            "Define a Pydantic model (subclassing BaseModel) describing the expected fields, "
            "then declare a parameter in your path operation function typed as that model. "
            "FastAPI reads and validates the JSON request body against it automatically."
        ),
        "ground_truth_contexts": [
            "To declare a **request** body, you use Pydantic models with all their power and benefits.",
            "First, you need to import `BaseModel` from `pydantic`",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q007",
        "question": "How do I add extra validation and metadata to a single field inside a Pydantic model used as a request body?",
        "ground_truth_answer": (
            "Use Field() from pydantic as the field's value, e.g. "
            "price: float = Field(gt=0, description=\"The price must be greater than zero\"). "
            "Field() supports the same kinds of validation arguments as Query/Path (gt, max_length, etc.) plus metadata like title and description."
        ),
        "ground_truth_contexts": [
            "price: float = Field(gt=0, description=\"The price must be greater than zero\")",
        ],
        "category": "direct",
        "difficulty": "medium",
    },
    {
        "id": "q008",
        "question": "By default, does FastAPI's Header parameter expect header names with underscores or hyphens in the incoming HTTP request?",
        "ground_truth_answer": (
            "Incoming HTTP headers use hyphens (e.g. user-agent), but you declare the Python "
            "parameter with underscores (e.g. user_agent). By default, Header automatically "
            "converts the parameter name's underscores to hyphens to match against the actual "
            "header, so you don't have to do that translation yourself."
        ),
        "ground_truth_contexts": [
            "by default, `Header` will convert the parameter names characters from underscore (`_`) to hyphen (`-`) to extract and document the headers.",
            "So, you can use `user_agent` as you normally would in Python code",
        ],
        "category": "direct",
        "difficulty": "medium",
    },
    {
        "id": "q009",
        "question": "What does the response_model parameter on a path operation do?",
        "ground_truth_answer": (
            "It declares the shape of the response data that FastAPI should validate, filter, "
            "document, and serialize the returned value against -- independent of the actual "
            "type the path operation function returns internally."
        ),
        "ground_truth_contexts": [
            "You can declare the type used for the response by annotating the *path operation function* return type.",
        ],
        "category": "direct",
        "difficulty": "medium",
    },
    {
        "id": "q010",
        "question": "How do I make a path operation return HTTP status code 201 instead of the default?",
        "ground_truth_answer": (
            "Pass status_code as a parameter of the decorator method itself (not the function), "
            "e.g. @app.post(\"/items/\", status_code=201)."
        ),
        "ground_truth_contexts": [
            "@app.post(\"/items/\", status_code=201)\nasync def create_item(name: str):",
            "Notice that `status_code` is a parameter of the \"decorator\" method (`get`, `post`, etc). Not of your *path operation function*",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q011",
        "question": "How do I accept a file upload in a FastAPI endpoint?",
        "ground_truth_answer": (
            "Import File and UploadFile from fastapi, and declare a parameter typed as "
            "UploadFile in your path operation function, e.g. "
            "async def create_upload_file(file: UploadFile): return {\"filename\": file.filename}."
        ),
        "ground_truth_contexts": [
            "Import `File` and `UploadFile` from `fastapi`",
            "async def create_upload_file(file: UploadFile):\n    return {\"filename\": file.filename}",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q012",
        "question": "How do I return a custom HTTP error (like a 404) with a specific message from a path operation?",
        "ground_truth_answer": (
            "Raise an HTTPException imported from fastapi, passing status_code and detail, "
            "e.g. raise HTTPException(status_code=404, detail=\"Item not found\")."
        ),
        "ground_truth_contexts": [
            "To return HTTP responses with errors to the client you use `HTTPException`.",
            "from fastapi import FastAPI, HTTPException",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q013",
        "question": "How do I schedule a function to run in the background after a response has already been sent to the client?",
        "ground_truth_answer": (
            "Declare a parameter of type BackgroundTasks in your path operation function, then "
            "call background_tasks.add_task(your_function, *args, **kwargs) inside it. FastAPI "
            "runs the task after sending the response."
        ),
        "ground_truth_contexts": [
            "background_tasks.add_task(write_notification, email, message=\"some notification\")",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    {
        "id": "q014",
        "question": "What do I need to install to use TestClient for testing a FastAPI app, and what naming convention should test functions follow?",
        "ground_truth_answer": (
            "Install httpx. Test function names should start with test_, following standard "
            "pytest convention."
        ),
        "ground_truth_contexts": [
            "To use `TestClient`, first install [`httpx`](https://www.python-httpx.org).",
            "Create functions with a name that starts with `test_` (this is a standard `pytest` convention).",
        ],
        "category": "direct",
        "difficulty": "easy",
    },
    # ---------------------------------------------------------------- multi_hop (6)
    {
        "id": "q015",
        "question": "Is CORS a standalone FastAPI feature, or is it implemented as a type of middleware -- and what does that mean for how you enable it?",
        "ground_truth_answer": (
            "CORS support is implemented as middleware: you enable it by importing "
            "CORSMiddleware from fastapi.middleware.cors and adding it to your app with "
            "app.add_middleware(CORSMiddleware, ...), the same general mechanism used for any "
            "other middleware, configured with CORS-specific parameters like allow_origins."
        ),
        "ground_truth_contexts": [
            "You can configure it in your **FastAPI** application using the `CORSMiddleware`.",
            "You can add middleware to **FastAPI** applications.",
        ],
        "category": "multi_hop",
        "difficulty": "medium",
    },
    {
        "id": "q016",
        "question": "How do class-based dependencies relate to function-based dependencies in FastAPI's dependency injection system -- do they work the same way?",
        "ground_truth_answer": (
            "Yes -- a class is usable as a dependency the same way a function is, because "
            "calling a class (to create an instance) uses the same syntax as calling a "
            "function. When you write Depends(CommonQueryParams), FastAPI calls "
            "CommonQueryParams the same way it would call a dependency function, creating an "
            "instance and injecting it as the parameter."
        ),
        "ground_truth_contexts": [
            "You might notice that to create an instance of a Python class, you use that same syntax.",
            "**FastAPI** calls the `CommonQueryParams` class. This creates an \"instance\" of that class and the instance will be passed as the parameter `commons` to your function.",
        ],
        "category": "multi_hop",
        "difficulty": "medium",
    },
    {
        "id": "q017",
        "question": "Does the basic OAuth2PasswordBearer setup shown in the first FastAPI security tutorial provide complete login security by itself, or is more work needed for a real JWT-based login flow?",
        "ground_truth_answer": (
            "The basic OAuth2PasswordBearer setup from the first tutorial only extracts a token "
            "from the request's Authorization header and shows it in the docs UI -- it "
            "explicitly does not verify the token's validity. A real, complete flow (like the "
            "JWT-based one) requires additional work: actually generating, signing, and "
            "verifying tokens against a user store."
        ),
        "ground_truth_contexts": [
            "We are not verifying the validity of the token yet, but that's a start already.",
            "So, in just 3 or 4 extra lines, you already have some primitive form of security.",
        ],
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "id": "q018",
        "question": "Can I accept both file uploads and regular form fields in the same FastAPI endpoint, and what do I need to import?",
        "ground_truth_answer": (
            "Yes. Import File, Form, and UploadFile all from fastapi, and declare parameters "
            "for each in the same path operation function -- form fields use Form(...) and "
            "uploaded files use UploadFile/File(...)."
        ),
        "ground_truth_contexts": [
            "from fastapi import FastAPI, File, Form, UploadFile",
        ],
        "category": "multi_hop",
        "difficulty": "medium",
    },
    {
        "id": "q019",
        "question": "By default, if a path operation function takes a single Pydantic model parameter for the request body, what JSON shape does FastAPI expect -- and how would you change that to expect the model nested under a key?",
        "ground_truth_answer": (
            "By default, with a single body parameter, FastAPI expects the model's fields "
            "directly at the top level of the JSON body. To instead expect the JSON nested "
            "under a key named after the parameter (e.g. {\"item\": {...}}), wrap the "
            "parameter's type with Body(embed=True), e.g. item: Annotated[Item, Body(embed=True)]."
        ),
        "ground_truth_contexts": [
            "By default, **FastAPI** will then expect its body directly.",
            "item: Annotated[Item, Body(embed=True)]",
        ],
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "id": "q020",
        "question": "In the bigger-applications pattern, if an APIRouter is created with prefix=\"/items\" and tags=[\"items\"], and a path operation inside it is defined as @router.get(\"/\"), what full path and tag will that operation end up with once included in the main app?",
        "ground_truth_answer": (
            "The full path is the router's prefix plus the operation's own path -- \"/items\" + "
            "\"/\" -> \"/items/\". The operation is also tagged \"items\", inherited from the "
            "router's tags."
        ),
        "ground_truth_contexts": [
            "* Path `prefix`: `/items`.\n* `tags`: (just one tag: `items`).",
            "router = APIRouter(\n    prefix=\"/items\",\n    tags=[\"items\"],",
        ],
        "category": "multi_hop",
        "difficulty": "medium",
    },
    # ---------------------------------------------------------------- unanswerable (5)
    {
        "id": "q021",
        "question": "How do I implement a WebSocket endpoint in FastAPI, including handling incoming and outgoing messages?",
        "ground_truth_answer": (
            "Not answerable from this documentation set -- WebSockets is only mentioned once, "
            "as a bullet point in FastAPI's feature list, with no procedural explanation of how "
            "to implement one."
        ),
        "ground_truth_contexts": [],
        "category": "unanswerable",
        "difficulty": "medium",
    },
    {
        "id": "q022",
        "question": "How do I serve a directory of static files, like images or CSS, from a FastAPI application?",
        "ground_truth_answer": (
            "Not answerable from this documentation set -- static file serving (e.g. "
            "StaticFiles) is not mentioned anywhere in the indexed pages."
        ),
        "ground_truth_contexts": [],
        "category": "unanswerable",
        "difficulty": "medium",
    },
    {
        "id": "q023",
        "question": "How do I set up a GraphQL endpoint in a FastAPI app using Strawberry?",
        "ground_truth_answer": (
            "Not answerable from this documentation set -- GraphQL/Strawberry integration is "
            "only mentioned as a bullet point in the feature list and in one aside about POST "
            "usage; there's no setup or implementation guidance in the indexed pages."
        ),
        "ground_truth_contexts": [],
        "category": "unanswerable",
        "difficulty": "medium",
    },
    {
        "id": "q024",
        "question": "How do I add rate limiting to a FastAPI endpoint to prevent clients from making too many requests?",
        "ground_truth_answer": (
            "Not answerable from this documentation set -- rate limiting or throttling is not "
            "covered anywhere in the indexed pages."
        ),
        "ground_truth_contexts": [],
        "category": "unanswerable",
        "difficulty": "easy",
    },
    {
        "id": "q025",
        "question": "How do I cache a FastAPI endpoint's response so that repeated identical requests don't recompute it on the server?",
        "ground_truth_answer": (
            "Not answerable from this documentation set -- the only caching-related mention is "
            "the CORS `max_age` parameter, which controls how long a browser caches a CORS "
            "preflight response, not general server-side response caching."
        ),
        "ground_truth_contexts": [
            "`max_age` - Sets a maximum time in seconds for browsers to cache CORS responses. Defaults to `600`.",
        ],
        "category": "unanswerable",
        "difficulty": "hard",
    },
    # ---------------------------------------------------------------- adversarial (7)
    {
        "id": "q026",
        "question": "If I call app.add_middleware(CORSMiddleware, ...) and then later add a custom logging middleware with @app.middleware(\"http\"), which one runs first when a request comes in?",
        "ground_truth_answer": (
            "The custom logging middleware (added second/last) runs first on the request path. "
            "Middlewares form a stack where the last one added is the outermost, and the "
            "outermost middleware runs first for incoming requests."
        ),
        "ground_truth_contexts": [
            "The last middleware added is the *outermost*, and the first is the *innermost*.",
            "On the request path, the *outermost* middleware runs first.",
        ],
        "category": "adversarial",
        "difficulty": "hard",
    },
    {
        "id": "q027",
        "question": "If a path parameter is declared as item_id: Annotated[int, Path(gt=0)], can a client omit item_id from the URL and still have the request succeed?",
        "ground_truth_answer": (
            "No. Path parameters are always required because they're part of the URL path "
            "itself -- there's no way to 'omit' a path parameter and still match the route. "
            "The gt=0 validation doesn't change that; it only restricts which values are valid "
            "when the parameter is present."
        ),
        "ground_truth_contexts": [
            "item_id: Annotated[int, Path(title=\"The ID of the item to get\", gt=0, le=1000)]",
        ],
        "category": "adversarial",
        "difficulty": "hard",
    },
    {
        "id": "q028",
        "question": "If a query parameter is declared with no default value at all (e.g. needy: str), is it optional?",
        "ground_truth_answer": (
            "No, it's required. In FastAPI, a query parameter without any declared default "
            "value is required -- only parameters with a default value (including None) are "
            "treated as optional."
        ),
        "ground_truth_contexts": [
            "But when you want to make a query parameter required, you can just not declare any default value",
            "Here the query parameter `needy` is a required query parameter of type `str`.",
        ],
        "category": "adversarial",
        "difficulty": "medium",
    },
    {
        "id": "q029",
        "question": "When using a class like CommonQueryParams as a dependency via Depends(CommonQueryParams), do you need to manually instantiate CommonQueryParams() yourself somewhere in your code?",
        "ground_truth_answer": (
            "No. FastAPI calls the class for you and creates the instance automatically; you "
            "don't need to instantiate it manually anywhere."
        ),
        "ground_truth_contexts": [
            "**FastAPI** calls the `CommonQueryParams` class. This creates an \"instance\" of that class and the instance will be passed as the parameter `commons` to your function.",
        ],
        "category": "adversarial",
        "difficulty": "medium",
    },
    {
        "id": "q030",
        "question": "Does the basic OAuth2PasswordBearer setup from FastAPI's first security tutorial actually verify that a submitted token is valid?",
        "ground_truth_answer": (
            "No -- the documentation explicitly says that setup is not yet verifying the "
            "validity of the token; it calls it only 'some primitive form of security' at that "
            "stage."
        ),
        "ground_truth_contexts": [
            "We are not verifying the validity of the token yet, but that's a start already.",
        ],
        "category": "adversarial",
        "difficulty": "medium",
    },
    {
        "id": "q031",
        "question": "If a client sends an HTTP header named X-Custom-Header, and I declare a FastAPI parameter as x_custom_header: Annotated[str, Header()] with default settings, will FastAPI correctly pick up that header's value?",
        "ground_truth_answer": (
            "Yes. By default, Header automatically converts the Python parameter name's "
            "underscores to hyphens (x_custom_header -> X-Custom-Header) to match against the "
            "incoming header name, so no extra configuration is needed for this to work."
        ),
        "ground_truth_contexts": [
            "by default, `Header` will convert the parameter names characters from underscore (`_`) to hyphen (`-`) to extract and document the headers.",
        ],
        "category": "adversarial",
        "difficulty": "medium",
    },
    {
        "id": "q032",
        "question": "Does Field() only add validation and documentation metadata to a Pydantic model attribute, or can it also set the attribute's default value?",
        "ground_truth_answer": (
            "Field() can also set the default value -- via its `default` argument, e.g. "
            "description: str | None = Field(default=None, title=\"...\", max_length=300). It's "
            "not limited to metadata/validation only."
        ),
        "ground_truth_contexts": [
            "description: str | None = Field(\n        default=None, title=\"The description of the item\", max_length=300\n    )",
        ],
        "category": "adversarial",
        "difficulty": "medium",
    },
    # ---------------------------------------------------------------- edge_case (3)
    {
        "id": "q033",
        "question": "cors?",
        "ground_truth_answer": (
            "CORS (Cross-Origin Resource Sharing) refers to when a frontend running in a "
            "browser communicates with a backend on a different origin (different protocol, "
            "domain, or port). FastAPI supports it via CORSMiddleware."
        ),
        "ground_truth_contexts": [
            "[CORS or \"Cross-Origin Resource Sharing\"] refers to the situations when a frontend running in a browser has JavaScript code that communicates with a backend, and the backend is in a different \"origin\" than the frontend.",
        ],
        "category": "edge_case",
        "difficulty": "medium",
    },
    {
        "id": "q034",
        "question": "How do I register a route in FastAPI using Flask's @app.route(\"/path\", methods=[\"GET\"]) decorator syntax?",
        "ground_truth_answer": (
            "FastAPI doesn't use Flask's @app.route(methods=[...]) syntax. Instead, it uses "
            "method-specific decorators like @app.get(\"/path\") or @app.post(\"/path\") -- there "
            "is no equivalent generic @app.route with a methods= list in FastAPI's documented "
            "API."
        ),
        "ground_truth_contexts": [
            "@app.get(\"/\")\nasync def root():",
        ],
        "category": "edge_case",
        "difficulty": "hard",
    },
    {
        "id": "q035",
        "question": "For a path parameter validated with ge=1, what values does it NOT allow?",
        "ground_truth_answer": (
            "It does not allow values less than 1 (i.e. 0, negative numbers, or anything below "
            "1) -- ge=1 requires the value to be greater than or equal to 1."
        ),
        "ground_truth_contexts": [
            "Here, with `ge=1`, `item_id` will need to be an integer number \"greater than or equal\" to `1`.",
        ],
        "category": "edge_case",
        "difficulty": "medium",
    },
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for pair in QA_PAIRS:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {len(QA_PAIRS)} QA pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
