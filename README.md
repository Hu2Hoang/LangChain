## Langchain

**Xử lý tài liệu PDF**

Sử dụng function `PyPDFLoader()` và `DirectoryLoader()` load tất cả tài liệu định dạng pdf chuẩn bị xử lý

    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

[Document Loader](https://python.langchain.com/docs/integrations/document_loaders) 

**Chia văn bản thành các phần nhỏ**

Dữ liệu sau khi load sẽ được xử lý để lấy ra các thông tin có ý nghĩa, sau đó được chia ra từng phần nhỏ trước khi chuyển đến bước tiếp theo.

Sử dụng `text_splitter` để chia các tài liệu văn bản thành các phần nhỏ hơn, có kích thước cố định để dùng cho việc nhúng và xử lý.

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

* `chunk_size` : Xác định kích thước tối đa của mồi phần nhỏ khi chia văn bản
* `chunk_overlap` : Xác định phần chồng chéo giữa 2 chunk_size liên tiếp khi chia văn bản.
  
## VectorDatabase
VectorDatabase được sử dụng để lưu trữ các biểu diễn vector của các phần văn bản. Mục đích của việc này có thể là để hỗ trợ các thao tác tìm kiếm và truy xuất thông tin từ các tài liệu văn bản dựa trên các tính năng của chúng

**Chuyển đổi dữ liệu sang dạng vector**


 Sử dụng `GPT4AllEmbeddings` để tạo một mô hình nhúng văn bản bằng biến thể của mạng nơ-ron GPT (Generative Pre-trained Transformer) nhằm biểu diễn các đoạn văn bản dưới dạng các vectơ có chiều cao trong không gian vector.

    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        
**Tạo, lưu VectorDatabase**

Sử dụng thư viện [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) để tạo và quản lý một cơ sở dữ liệu vector từ các đoạn văn bản đã được nhúng. Dữ liệu sau khi chuyển đổi sang vector được lưu vào vector store.

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)

### FASISS

## Dựng model hỏi đáp

* Load model từ file gguf đã tải bằng cách sử dụng hàm load_llm.

        def load_llm(model_file):
            llm = CTransformers(
                model=model_file,
                model_type="llama",
                max_new_tokens=4096, #Số lương token tối đa mà mô hình có thể sinh ra
                temperature=0.01 #Kiểm soát độ đa dạng của câu trả lời
            )
        return llm

* Tạo promt

        def creat_prompt(template):
            prompt = PromptTemplate(template = template, input_variables=["context", "question"])
            return prompt

* Tạo, đọc dữ liệu từ VectorDatabase

        def read_vectors_db():
            embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
            db = FAISS.load_local(vector_db_path, embedding_model,allow_dangerous_deserialization=True)
            return db

* Tạo chuỗi xử lý câu hỏi và câu trả lời
  
        def create_qa_chain(prompt, llm, db):
            llm_chain = RetrievalQA.from_chain_type(
                llm = llm,
                chain_type= "stuff",
                retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
                return_source_documents = False,
                chain_type_kwargs= {'prompt': prompt}
            )
            return llm_chain

* Tạo câu hỏi:
 
        question = "Nghị định 13 yêu cầu gì với dữ liệu"
        response = llm_chain.invoke({"query": question})
        print(response)

* Kết quả:

        {'query': 'Nghị định 13 yêu cầu gì với dữ liệu', 'result': '\nNghị định số 13 về bảo vệ dữ liệu cá nhân là một văn bản pháp lý do chính phủ ban hành, nhằm quy định việc bảo vệ dữ liệu cá nhân và trách nhiệm của các bên liên quan. Nó đề cập đến nhiều khía cạnh khác nhau trong quản lý và xử lý dữ liệu, đảm bảo an ninh và quyền riêng tư cho thông tin cá nhân.\n\nNghị định này áp dụng cho tất cả các tổ chức, cơ sở và cá nhân tham gia thu thập, xử lý hoặc lưu trữ dữ liệu cá nhân. Nó cũng ảnh hưởng đến hoạt động kinh doanh của các công ty vì nó đòi hỏi họ phải tuân theo các quy tắc và tiêu chuẩn cụ thể để bảo vệ thông tin được giữ gìn bí mật.\n\nNghị định 13 bao gồm một số điều khoản chính sau đây:\n\n1. Bảo đảm pháp luật quy định về dữ liệu của chủ thể có liên quan\n2. Quy định về tính minh bạch, trách nhiệm giải trình trong lĩnh vực bảo vệ dữ liệu\n3. Về cách thức xử lý dữ liệu cá nhân\n4. Quyền và nghĩa vụ của các bên liên quan như chủ thể, đối tượng trưng bày viên chức năng lực lượng tử thi hành khách thể chế độ chính sách nước sử dụng cụ thể chế độ nhạy cảm ứng'}