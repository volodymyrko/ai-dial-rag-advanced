from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


#TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope

#TODO:
# Provide structured system prompt, with RAG Context and User Question sections.

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


#TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)
# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml


dimensions = 1536
top_k = 10
score_threshold = 0.5
text_split_chunk_size = 300
text_split_overelap = 50
db_config = {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}

embeddings_client = DialEmbeddingsClient('text-embedding-3-small-1', API_KEY)
chat_client = DialChatCompletionClient('gpt-4o', API_KEY)

text_processor = TextProcessor(embeddings_client, db_config)
text_processor.process_text_file(
	file_name='task/embeddings/microwave_manual.txt',
	chunk_size=text_split_chunk_size,
	overlap=text_split_overelap,
	dimensions=dimensions,
	truncate=True
)


print('Enter your question or type \'exit\'')

while True:
	user_question = input("\n> ").strip()

	if user_question == 'exit':
		print('Exiting...')
		break

	context_parts = text_processor.search(user_question, SearchMode.COSINE_DISTANCE, top_k, score_threshold, dimensions)
	system_msg = Message(
		role=Role.SYSTEM,
		content=SYSTEM_PROMPT
	)

	context = '\n\n'.join(context_parts)
	user_msg = Message(
		role=Role.USER,
		content=USER_PROMPT.format(context=context, query=user_question)
	)
	ai_reposne = chat_client.get_completion(messages=[system_msg, user_msg], print_request=False)
	print('response:', ai_reposne.content)

