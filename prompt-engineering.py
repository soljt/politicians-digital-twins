from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import query_kb

from langchain_community.vectorstores import FAISS

# Setup OpenAI API (or you can use any other LLM provider)
# Initialize the OpenAI LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# make the prompt templates
system_prompt = """
You are a digital twin of {candidate}, who is {role}. You are taking questions from an audience.
You are provided with information from your own publications relevant to the question you are asked, and you respond based on this information. It is delimited by 'retrieval:' and each separate passge is delimited by a numeral like '1.'.
If the question is something you are willing to discuss, respond with an answer. If not, respond with a polite refusal and a redirection specifically to computational social science.
"""

rag_retrieval = """
retrieval: \n{statement}
question: {question}
"""

one_shot_query = """
retrieval:
1. The emergence of computational tools and increased data availability has given rise to a new domain known as Computational Diplomacy, attracting attention toward digitally enhanced participatory democracy. This area explores how digital technologies can augment democratic processes, emphasizing civic engagement, governance transparency, and inclusivity.

Integrating digital democracy into political frameworks can enhance citizen participation, making government processes more accessible and efficient. Key themes include enabling online voting, petitioning, digital campaigning, and deliberative forums. However, challenges like misinformation, digital divides, and transparency issues must be addressed to ensure equitable technology use and to avoid reinforcing existing power disparities.

2. Finally, future research must explore how digital assistance tools can be designed to support democratic processes while preventing malicious misuse. The goal is to create systems that promote informed decision-making, ensuring they are resilient to external manipulations while encouraging broad participation across varying demographic segments.

In sum, the authors envision a digitally transformed democratic landscape, where technology serves as a bridge for engagement and inclusion, reshaping governance for a more participatory and equitable society.

3. The authors advocate for a “democracy by design” approach, implementing digital systems that prioritize democratic values such as privacy, inclusion, and equity. This value-based engineering should drive the development of digital citizenship frameworks to ensure that every individual can access and benefit from political processes.

In addressing misinformation, strategies such as digital literacy and fact-checking campaigns are pivotal. Additionally, maintaining diverse perspectives is beneficial for innovation and democratic resilience, necessitating interaction networks that encourage knowledge sharing and collective intelligence.

question: What does digital democracy mean?
"""

one_shot_response = """
Good question. Well, digital democracy refers to the integration of digital technologies into democratic processes to enhance citizen participation, governance transparency, and inclusivity. It's about leveraging computational tools and the data that's so pervasive in today's day and age to create a more participatory and fair political landscape. Digital democracy wants to make government processes more accessible and efficient by incorporating elements like online voting, digital campaigning, and deliberative forums.
Of course, such a thing doesn't come without challenges. Misinformation, digital divides, and transparency issues are a few that come to mind. These types of issues must be addressed to make sure that technology serves as a bridge for engagement and inclusion, rather than reinforcing existing power disparities. The goal is to design digital systems that uphold democratic values like privacy, inclusion, and fairness. Hopefully, this results in a resilient, informed, and participatory society.
"""

off_topic_query = """
retrieval:

question: how can i cook a perfect chicken breast?
"""

off_topic_response = """
I appreciate your interest in cooking, but I don't think it's really relevant to my work. Let's try not to let our discussion wander.
"""

previous_query = """
retrieval: \n{previous_retrieval}
question: {previous_question}
"""

previous_response = """
{previous_output}
"""

previous_retrieval = "*empty*\n"
previous_question = "*empty*\n"
previous_output = "*empty*\n"

RAG_prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("system", "Here are some example interactions:\n"),
    ("human", off_topic_query),
    ("ai", off_topic_response),    
    ("human", one_shot_query),
    ("ai", one_shot_response),
    ("system", "The following is the most recent exchange between you and the user:\n"),
    ("human", previous_query),
    ("ai", previous_response),
    ("system", "Now you are asked:\n"),
    ("human", rag_retrieval),
    ("system", "Respond in a conversational manner to the question")
])

irrelevant_prompt = PromptTemplate(
    input_variables=["question"], 
    template="A user has asked you: {question}. This is not something you are concerned with answering. Express to the user in your own words that you are not willing to discuss this topic."
)


####################################################
# choose the db to use
db = FAISS.load_local("faiss_index_text", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 3})

# Setup the conversation
candidate = "Dirk Helbing"
role = "a professor of computational social science at ETH Zurich"
print(f"Dirk: Hello! I'm Dirk Helbing, a professor of computational social science at ETH Zurich. What would you like to ask?")
user_question = input("You: ")

# Main conversation loop
while user_question != "exit":
    # get relevant passages from the database
    results = retriever.invoke(user_question)

    # if question out of scope or no relevant passages retrieved, respond with irrelevant prompt
    # if results == []:
    #     response = llm.invoke(irrelevant_prompt.format(question=user_question))
    # else:

    #################### New pipeline ####################
    passages = "\n".join([f"{idx + 1}. {result.page_content}\n" for idx, result in enumerate(results)])
    prompt = RAG_prompt.format_messages(candidate=candidate, role=role, statement=passages, question=user_question, previous_retrieval=previous_retrieval, previous_question=previous_question, previous_output=previous_output)
    print(f"\n{'-'*80}")
    llm_in = "".join(f"{item.type.upper()}: {item.content}" for item in prompt)
    print(llm_in)
    print(f"\n{'-'*80}")
    # get response from LLM
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        response = llm.invoke(irrelevant_prompt.format(question=user_question))
    ######################################################

    # store last response for later use
    previous_retrieval = passages
    previous_question = user_question
    previous_output = response.content

    # print response and continue conversation
    print(f"Dirk: {response.content}")
    user_question = input("You: ")
