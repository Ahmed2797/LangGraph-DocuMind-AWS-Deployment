import asyncio
from src.pipeline import pipeline


def print_graph(app):
    """
    Save the LangGraph workflow as a PNG image.
    """
    print("\n" + "=" * 80)
    print("LANGGRAPH WORKFLOW")
    print("=" * 80)

    graph = app.get_graph()
    # Terminal visualization
    # print(app.get_graph().draw_ascii())

    # Mermaid source when needed
    # print(app.get_graph().draw_mermaid())

    png_bytes = graph.draw_mermaid_png()

    output_path = "langgraph_workflow-sample-v2.png"

    with open(output_path, "wb") as f:
        f.write(png_bytes)

    print(f"Graph saved to {output_path}")
    print("=" * 80)


async def main():
    """Initialize the pipeline and visualize its workflow."""
    async with pipeline("test") as app:
        print_graph(app)


if __name__ == "__main__":
    asyncio.run(main())

### pip install grandalf