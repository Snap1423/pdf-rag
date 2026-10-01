from rag import ask_question

print("=" * 50)
print("       PDF QUESTION ANSWERING SYSTEM")
print("=" * 50)

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if not query.strip():
        print("Please enter a question.")
        continue

    print("\nThinking...")

    try:
        answer, sources = ask_question(query)

        print("\nAnswer:")
        print(answer)

        print("\nSource Pages:")

        for page in sources:
            print(f"- Page {page}")
            
    except Exception as e:
        print(f"\nError: {e}")