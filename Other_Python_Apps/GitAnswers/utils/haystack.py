import streamlit as st
import streamlit
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore
from haystack.nodes import JoinDocuments
from haystack import Pipeline
from haystack.nodes import PreProcessor
from haystack.schema import Answer
from haystack.nodes import TextConverter
from haystack.nodes import PDFToTextConverter
from haystack.nodes import DocxToTextConverter
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
#Use this file to set up your Haystack pipeline and querying
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents, convert_files_to_docs, tika_convert_files_to_docs
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
import logging
from docx import *
from pathlib import Path
from elasticsearch import Elasticsearch
import os
import tika
tika.TikaClientOnly = True
from tika import parser
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
question = ''
params = {}
# document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="pica8qsg", embedding_dim=384)
if os.path.exists("./faiss_document_store.db"):
    print ("THERE BE WHALES HERE!")
    os.remove("./faiss_document_store.db")
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", embedding_dim=384, index="pica8docs3a", similarity="cosine", embedding_field="question_emb", sql_url="sqlite:///faiss_document_store.db")
    #document_store = FAISSDocumentStore.load(index_path="my_faiss", config_path="my_faiss.json")
    # document_store = FAISSDocumentStore.load(index_path="my_faiss")
    # # document_store = FAISSDocumentStore(faiss_config_path="./my_faiss.json", faiss_index_path="./my_faiss")
    doc_dir = "./newreadmes/"
    dicts = convert_files_to_docs(dir_path=doc_dir)
    print(dicts[:3])
    document_store.write_documents(dicts)
    document_store.save(index_path="./faissshift.index", config_path="./faiss.json")
    document_store.save("my_faiss")
    
else:
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", embedding_dim=384, index="pica8docs3a", similarity="cosine", embedding_field="question_emb", sql_url="sqlite:///faiss_document_store.db")
    doc_dir = "./newreadmes/"
    dicts = convert_files_to_docs(dir_path=doc_dir)
    print(dicts[:3])
    # from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
    # document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="pica8docs")
    # converter = PDFToTextConverter(remove_numeric_tables=True)
    # #doc_pdf = converter.convert(file_path="data/preprocessing_tutorial/bert.pdf", meta=None)
    # doc = converter.convert(file_path=filename, meta={'name':str(filename)})
    # processor = PreProcessor(
    #     clean_empty_lines=True,
    #     clean_whitespace=True,
    #     clean_header_footer=True,
    #     split_by="word",
    #     split_length=200,
    #     split_respect_sentence_boundary=True,
    #     split_overlap=0
    #     )
    # docs = processor.process(doc)
    # print (docs)
    # document_store.write_documents(docs)
    document_store.write_documents(dicts)
    document_store.save(index_path="./faissshift.index", config_path="./faiss.json")
    document_store.save("my_faiss")
    
    #   result = list(Path("/home/ntrieber/Downloads/Pica8_Docs").rglob("*.[pP][dD][fF]"))

    #   filename = "/home/ntrieber/Downloads/Pica8_Docs/AmpConGuide.pdf"

    #   print ('Working on File: ' + str(filename))
    #   #converter = PDFToTextConverter(remove_numeric_tables=False, valid_languages=["de","en"])
    #   converter = PDFToTextConverter(remove_numeric_tables=True)
    #   #doc_pdf = converter.convert(file_path="data/preprocessing_tutorial/bert.pdf", meta=None)
    #   doc = converter.convert(file_path=filename, meta={'name':str(filename)})


    #   processor = PreProcessor(
    #       clean_empty_lines=True,
    #       clean_whitespace=True,
    #       clean_header_footer=True,
    #       split_by="word",
    #       split_length=200,
    #       split_respect_sentence_boundary=True,
    #       split_overlap=0
    #     )
    #   docs = processor.process(doc)
    #   print (docs)
    #   document_store.write_documents(docs)
    #   document_store.save(index_path="./faissshift.index", config_path="./faiss.json")
    #   document_store.save("my_faiss")

answers = ''
# cached to make index and models load only at start
@st.cache_resource(show_spinner=True)
def start_haystack(_ds, question, answers):
    print("Received question: " + str(question))
    #retriever = ElasticsearchRetriever(document_store=_ds, top_k=10, )
    reader = FARMReader(model_name_or_path="deepset/deberta-v3-base-injection", use_gpu=True)
    embedding_retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2", use_gpu=True, top_k=10)
    #bm25_retriever = BM25Retriever(document_store=document_store, top_k=10, custom_query=question)
    #retriever = EmbeddingRetriever(
    #document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1", use_gpu=True, )
    document_store.update_embeddings(embedding_retriever)
    # pipe = Pipeline()
    # pipe.add_node(component=bm25_retriever, name="Retriever", inputs=["Query"])
    # pipe.add_node(component=reader, name="Reader", inputs=["Retriever"])
    pipex = ExtractiveQAPipeline(reader, embedding_retriever)
    prediction = pipex.run(
    query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
    print ('PIPEXED!')
    print("Received question" + str(question))
    print_answers(prediction, details="all")
    theanswers=[x.to_dict() for x in prediction["answers"]]

    print ("PREDICTIONS: ")
    for i in range(len(theanswers)):
        print (theanswers[i])
    
    # print ("YOOOOOOO! " + answers[1])
    # print ("***************************************************************************")
    # answers = print_answers(prediction, details="all")
    print ("***************************************************************************")
    lfqa_prompt = PromptTemplate(
    prompt="""Synthesize a comprehensive answer from the following text for the given question.
                             Provide a clear and concise response that summarizes the key points and information presented in the text.
                             Your answer should be in your own words and be no longer than 50 words.
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
    output_parser=AnswerParser(),)

    prompt_node = PromptNode(model_name_or_path="google/flan-t5-large", default_prompt_template=lfqa_prompt)
    
    pipe = Pipeline()
    pipe.add_node(component=embedding_retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])
    output = pipe.run(query=question)
    print ("Here's What prompted me:")
    for prompt in output["answers"]:
        print ("Answer: " + prompt.answer)
        print ("*************************************************************")
        # for answer in answers:
    #     print ("The Answer: " + answer)
    # #Use this function to contruct a pipeline
    # pipeline = Pipeline()
    # # Initialize Sparse Retriever
    # bm25_retriever = BM25Retriever(document_store=document_store)
    # query = question
    # # Initialize embedding Retriever
   # )
    # document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

    # # Initialize Reader
    # reader = FARMReader(model_name_or_path="deepset/deberta-v3-base-injection", use_gpu=True)


    # #p_retrieval = DocumentSearchPipeline(bm25_retriever)

    # # res = p_retrieval.run(query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}})
    # # print_documents(res, max_text_len=200)

    # # p_extractive_premade = ExtractiveQAPipeline(reader=reader, retriever=bm25_retriever)
    # # res = p_extractive_premade.run(
    # #     query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    # # )
    # # print_answers(res, details="minimum")
    # #pipeline.add_node(component=bm25_retriever, name="ESRetriever1", inputs=["Query"])
    # # pipe.add_node(component=reader, name="QAReader", inputs=["ESRetriever1"])
    # #pipeline.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    # pipeline.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
    # # pipeline.add_node(
    # #     component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["EmbeddingRetriever", "ESRetriever1"]
    # # )
    # pipeline.add_node(component=reader, name="QAReader", inputs=["EmbeddingRetriever"])

    # # Uncomment the following to generate the pipeline image
    # # pipe.draw("pipeline_ensemble.png")

    # # Run pipeline
    # res = pipeline.run(
    #     query=question, params={"EmbeddingRetriever": {"top_k": 10}}
    # )
    # print_answers(res, details="maximum")
    return theanswers

# answers = start_haystack(document_store, question, answers)
# print (answers)

@st.cache_data(show_spinner=True)
def query(question):
    print("Received question: " + str(question))
    params = {}
    answersset1 = ''
    answersset1 = start_haystack(document_store, question, answers)
    print ("Here's what we found out: ")
    print ("***************************************************************************")
    for i in range(len(answersset1)):
        print (answersset1[i])
    
    # for key, value in results.items():
    #         print (key, ': ', value)
    return answersset1