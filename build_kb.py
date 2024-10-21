from sentence_transformers import SentenceTransformer
from knowledge_base import KnowledgeBase
import tqdm
import pickle
import passages

kb = [passage["content"] for passage in passages.passages]

# Dense retrieval using Sentence Transformers
model_embd = SentenceTransformer("bert-base-nli-mean-tokens").to("cuda:0")
kb_index_embd = KnowledgeBase(dim=768)

# Add each passage to the knowledge base
for passage_index, passage_embd in enumerate(tqdm.tqdm(kb)):
    kb_index_embd.add_item(model_embd.encode(passage_embd).squeeze(), passage_index)

# Save the knowledge base index to disk
with open("kb_index_embd.pkl", "wb") as f:
    pickle.dump(kb_index_embd, f)

print("Knowledge base index saved to disk.")

