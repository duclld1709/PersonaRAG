# Câu hỏi phỏng vấn: PersonaRAG (Research Assistant for Students)

Bộ 15 câu hỏi một nhà tuyển dụng (AI Engineer / LLM Engineer, Backend, Full-stack AI) có thể hỏi về
dự án này, kèm gợi ý trả lời bám theo code thật trong repo. Mỗi câu có ba phần: người phỏng vấn muốn
kiểm tra điều gì, gợi ý trả lời, và câu hỏi nối tiếp nên chuẩn bị.

> Mẹo chung: RAG là chủ đề rất phổ biến, nên nhà tuyển dụng sẽ tìm điểm **khác biệt**. Điểm khác biệt
> của dự án này: (1) mỗi kỹ thuật (hybrid, multi-query, rerank, cache, router) được **đo** trên một
> harness có thể tái lập, thay vì mặc định là có ích; (2) bạn **tự rút lại số liệu sai** của v1;
> (3) các bug tinh vi về cache, đo chi phí và async đã được tìm và sửa.

---

## Q1. Giới thiệu dự án trong 2 phút.

**Muốn kiểm tra:** khả năng tóm tắt, hiểu kiến trúc tổng thể.

**Gợi ý trả lời:**
- **Bài toán:** sinh viên có rất nhiều tài liệu (slide, paper, giáo trình), và ChatGPT thường bịa trên
  nền đó. PersonaRAG **chỉ trả lời từ tài liệu người dùng upload**, trích dẫn từng ý `[1]`, `[2]`, và nói
  thẳng "I could not find this in your materials" khi không có căn cứ.
- **Pipeline một lượt chat** (`backend/app/services/chat_service.py`): moderation → router (retrieve,
  chitchat hoặc refuse) → semantic cache (chỉ lượt đầu) → viết lại câu hỏi theo lịch sử → mở rộng
  thành 3 truy vấn → hybrid retrieval (BM25 + vector, fuse bằng RRF) → LLM rerank → sinh câu trả lời
  stream qua SSE → lưu cache → ghi usage và chi phí.
- **Stack:** FastAPI theo Clean Architecture, OpenAI (`gpt-4.1`, `gpt-4.1-mini`,
  `text-embedding-3-large`, moderation), Chroma, SQLite (WAL), frontend Next.js 14 + Tailwind.
- **Evaluation:** golden set 81 câu có **trích dẫn bằng chứng nguyên văn**, corpus 400 chunk, metric
  IR kèm bootstrap CI, LLM judge, CI chặn merge nếu BM25 Hit@5 < 0.80.

---

## Q2. Hybrid retrieval của bạn hoạt động thế nào? Tại sao dùng RRF?

**Muốn kiểm tra:** hiểu sâu retrieval, không chỉ gọi thư viện.

**Gợi ý trả lời (`backend/app/retrieval/hybrid.py`):**
- Với **mỗi truy vấn** (câu gốc cộng các biến thể từ multi-query), chạy cả BM25 lẫn vector search
  (mặc định MMR, `fetch_k = 20`, `lambda_mult = 0.5` để đa dạng hoá).
- **Reciprocal Rank Fusion:** mỗi document nhận `weight / (rrf_k + rank)`, với `rrf_k = 60`, trọng số
  vector 0.6 và BM25 0.4. Điểm được **cộng dồn qua mọi (truy vấn, retriever)**, nên chunk xuất hiện ở
  nhiều danh sách được đẩy lên.
- **Tại sao RRF:** điểm BM25 (không giới hạn) và cosine (khoảng −1 đến 1) **không cùng thang đo**, cộng
  thẳng là vô nghĩa. RRF chỉ dùng **thứ hạng**, nên không cần chuẩn hoá và bền với outlier.
- Đặt trọng số = 0 sẽ **tắt hẳn nhánh đó** (không gọi embedding), nên eval đo được BM25-only và
  dense-only bằng cùng một code.
- Khoá khử trùng lặp là `chunk_id` (ổn định), không phải nội dung.

**Nối tiếp:** "Vì sao BM25 vẫn cần khi đã có embedding?" Embedding yếu với từ khoá hiếm, tên riêng,
ký hiệu và mã ("L2 regularization", "AdamW"). BM25 khớp chính xác từ. Hai loại bổ trợ cho nhau.

---

## Q3. Reranker của bạn làm gì, và bug "0.3 × upstream" của v1 là gì?

**Muốn kiểm tra:** khả năng đọc kỹ code số học, hiểu thang đo.

**Gợi ý trả lời (`backend/app/retrieval/reranker.py`):**
- Reranker dùng model nhỏ (`gpt-4.1-mini`) với **structured output** (Pydantic schema
  `_RerankResponse`) chấm điểm từ 0 đến 10 cho từng ứng viên (mỗi đoạn cắt 800 ký tự), rồi giữ top-N.
- Điểm cuối = `0.7 × (LLM / 10) + 0.3 × (RRF / max_RRF)`. Phần RRF giúp phá thế hoà khi LLM chấm gần
  bằng nhau.
- **Bug của v1:** blend với **điểm RRF thô**. RRF chỉ khoảng 0.01 (ví dụ 0.6/61), nên `0.3 × 0.01` gần
  bằng 0 và trọng số 0.3 thực chất **không có tác dụng**. v2 chuẩn hoá RRF về [0, 1] trước khi blend.
  Test: `test_reranker_rescales_fusion_scores_before_blending`.
- **Tối ưu:** nếu số ứng viên ≤ top_n thì bỏ qua lời gọi API. **Chịu lỗi:** API lỗi thì quay về thứ tự
  trước rerank.

**Nối tiếp:** "Sao không dùng cross-encoder như Cohere hay bge-reranker?" Để giữ một nhà cung cấp, dùng
chung billing và prompt caching. Đánh đổi là chậm hơn và đắt hơn một cross-encoder nhỏ chạy local. Đây
là điểm nên đo trong ablation.

---

## Q4. Semantic cache của v1 sai ở đâu, và v2 sửa thế nào?

**Muốn kiểm tra:** tư duy về tính đúng đắn của cache. **Đây là câu "ăn điểm" nhất.**

**Gợi ý trả lời (`backend/app/retrieval/cache.py`):**
- v1 dùng cache **toàn cục** theo độ tương đồng cosine của câu hỏi, nên có ba lỗi:
  1. **Sai chế độ:** câu trả lời dạng *Quiz* bị trả cho yêu cầu *Explain* cùng câu hỏi.
  2. **Sai hội thoại:** câu nối tiếp "còn ý thứ hai thì sao?" khớp với câu trả lời của **hội thoại
     khác**.
  3. **Dữ liệu cũ:** trả câu trả lời trích dẫn tài liệu **đã bị xoá hoặc sửa**.
- Cách sửa của v2:
  - **Scope theo mode:** entry chỉ khớp request cùng `scope = mode`.
  - **Chỉ lượt đầu:** `use_cache = enable_semantic_cache and not history`.
  - **Vô hiệu hoá:** callback `_on_corpus_change` trong `bootstrap.py` gọi `cache.clear()` (và rebuild
    BM25) mỗi khi corpus thay đổi.
  - **Không cache câu "not found"**, vì sau khi người dùng upload tài liệu mới thì câu đó có thể trả
    lời được.
  - **Ngưỡng đọc trực tiếp** từ settings (`threshold` là callable), nên thanh trượt trên UI có tác dụng
    ngay. v1 chép giá trị lúc khởi động.
  - FIFO tối đa 256 entry, TTL 24 giờ, có `threading.Lock`.
- Mỗi điểm có một test trong `backend/tests/test_cache.py`.

**Nối tiếp:** "Cache quét tuyến tính O(n) có chậm không?" Với 256 entry thì không đáng kể so với một lời
gọi embedding. Lớn hơn thì dùng ANN index hoặc Redis vector search.

---

## Q5. Đo chi phí theo từng request thế nào? Vì sao phải dùng `ContextVar` và cộng tay ở phần stream?

**Muốn kiểm tra:** hiểu concurrency trong Python, observability cho LLM.

**Gợi ý trả lời (`backend/app/core/usage.py`):**
- **Vấn đề v1:** chỉ đếm lời gọi sinh câu trả lời cuối, bỏ qua router, rewrite, expand, rerank và
  embedding, nên chi phí bị báo thiếu.
- **v2:** mỗi request có một `UsageTracker`. Block `with tracking(tracker, "rerank"):` đặt một
  `ContextVar`. Các adapter OpenAI gọi `record()` sau mỗi lời gọi và tự cộng vào tracker đang hoạt
  động, gắn **stage**. Kết quả là chi phí chia theo từng giai đoạn (`cost_breakdown`) cho UI và eval.
- **Chi tiết tinh vi:** pipeline là một **generator** được `iterate_in_threadpool` đẩy từng bước, và mỗi
  bước có thể chạy trên một thread hay context khác nhau. **`ContextVar` không tồn tại qua các lần
  `yield`**, nên `tracking()` chỉ được bọc code đồng bộ, không bao quanh `yield`. Phần stream sinh câu
  trả lời được cộng **thủ công** bằng `tracker.add()` khi nhận usage ở chunk cuối (`_stream_llm`).
- Giá tính theo bảng `pricing.py`, có giá riêng cho **cached input token** (OpenAI prompt caching).
- Ghi DB **một dòng cho mỗi model**, nhóm theo `request_id`, để tổng hợp chi phí theo model chính xác.
  Test `test_usage_summary_counts_requests_not_rows` đảm bảo đếm request chứ không đếm dòng.

---

## Q6. Streaming SSE: v1 bị chặn event loop thế nào và bạn sửa ra sao?

**Muốn kiểm tra:** hiểu async/await trong FastAPI.

**Gợi ý trả lời (`backend/app/api/routes/chat.py`):**
- Pipeline gọi OpenAI và Chroma bằng **client đồng bộ (blocking)**. v1 chạy chúng trực tiếp trong async
  generator của SSE, tức **trên event loop**. Trong lúc một câu trả lời đang stream, mọi request khác
  (kể cả `/health`) đều bị treo.
- v2: `iterate_in_threadpool(iter(chat_service.stream_answer(...)))`, nên mỗi bước của generator chạy
  trên threadpool và event loop chỉ `await` kết quả.
- Định dạng SSE: `data: {json}\n\n`. Các event có kiểu: `status` (routing, retrieving, reranking,
  generating, cache_hit…), `delta` (token), `citations`, rồi `usage`/`latency_ms`/`cost_breakdown`, cuối
  cùng là `[DONE]`.
- Có **whitelist** key được phép gửi ra client (`_prepare_stream_payload`). Event nội bộ `contexts`
  (dành cho eval) không bị lộ.
- Lỗi: log đầy đủ ở server, client chỉ nhận **thông báo chung chung** (v1 gửi nguyên text exception, có
  thể lộ thông tin nội bộ).

**Nối tiếp:** "Sao không dùng `AsyncOpenAI`?" Cũng là một lựa chọn tốt, nhưng phải chuyển toàn bộ
pipeline sang async. Threadpool là thay đổi nhỏ nhất mà sửa đúng gốc lỗi.

---

## Q7. Vì sao `generate_answer` lại "tiêu thụ" `stream_answer`?

**Muốn kiểm tra:** thiết kế để tránh trùng lặp và lệch logic.

**Gợi ý trả lời:**
- v1 có **hai bản pipeline viết tay**: một cho stream, một cho REST và eval. Hai bản đã **lệch nhau**,
  và eval còn **ép mọi câu hỏi vào nhánh retrieve**, nên router và hành vi với câu ngoài phạm vi chưa
  bao giờ được đo.
- v2 chỉ có **một pipeline**. `generate_answer` lặp qua các event của `stream_answer` và gom `delta`,
  `citations`, `contexts`, `usage` thành `ChatGenerationResult`. Eval vì vậy đo **đúng code người dùng
  chạy**, kể cả router.
- Test `test_respond_endpoint_matches_stream` đảm bảo hai endpoint cho cùng kết quả.

---

## Q8. Ingestion "incremental reindex" hoạt động thế nào? Điểm yếu của nó là gì?

**Muốn kiểm tra:** hiểu thiết kế ingestion, và tự nhìn ra hạn chế.

**Gợi ý trả lời (`backend/app/ingestion/pipeline.py`, `splitters.py`):**
- Chia chunk bằng `RecursiveCharacterTextSplitter` (800 ký tự, overlap 150, ưu tiên cắt ở đoạn, dòng,
  rồi câu).
- **PDF:** trang được ngăn bằng `\f`, và chunk **không bao giờ vắt qua hai trang**, nên citation có số
  trang. v1 nối mọi trang thành một chuỗi nên mất số trang.
- Mỗi chunk có `chunk_id = "{document_id}:{index}"` và `chunk_checksum = sha1(content)`. Khi upload lại,
  so với chunk cũ trong Chroma: checksum và metadata giống thì **giữ nguyên**, khác thì xoá và thêm lại,
  chunk thừa thì xoá. Chỉ phần thay đổi bị embed lại, nên tiết kiệm chi phí embedding.
- Có thay đổi thì gọi `on_change`: rebuild BM25 và xoá semantic cache.
- **Điểm yếu (nên chủ động nói):** id dựa trên **vị trí**. Chèn một đoạn ở đầu tài liệu làm mọi index
  phía sau dịch chuyển, nên hầu hết chunk bị coi là "thay đổi" và embed lại. Cách tốt hơn là dùng
  **content-addressed id** (hash nội dung) hoặc cache embedding theo checksum. Ngoài ra BM25 được
  **rebuild toàn bộ** mỗi lần, tức O(N), chấp nhận được ở quy mô vài nghìn chunk.

---

## Q9. Bạn đánh giá chất lượng retrieval thế nào? Tại sao v1 sai?

**Muốn kiểm tra:** thiết kế evaluation cho RAG, kiến thức metric IR.

**Gợi ý trả lời (`backend/eval/`):**
- **v1 sai:** precision/recall tính theo **tên file** trên corpus chỉ có **2 file**, nên gần như chunk
  nào cũng "đúng".
- **v2:**
  - Corpus 400 chunk: 2 bài giảng tự viết cộng 14 phần của *Dive into Deep Learning*, tải ở commit cố
    định và kiểm SHA-256. Các chủ đề chồng nhau (overfitting, regularization, optimizer) đóng vai trò
    **distractor thực tế**.
  - 81 câu hỏi: 69 câu một nguồn, 6 câu nhiều chương, 4 câu ngoài phạm vi, 2 câu vi phạm liêm chính học
    thuật.
  - Mỗi câu có **trích dẫn bằng chứng nguyên văn**. Chunk được tính là liên quan nếu chứa bằng chứng
    (hoặc ≥ 60% liên tục của nó, cho trường hợp bằng chứng bị cắt giữa hai chunk; hàm `covers` trong
    `eval/dataset.py`). Có test kiểm tra mọi trích dẫn thực sự tồn tại trong corpus.
- **Metric:** Hit@k, Recall@k (tỉ lệ bằng chứng được phủ, quan trọng với câu hỏi nhiều ý), MRR, nDCG@k.
- **Bootstrap CI 95%:** với 75 câu in-scope, khoảng tin cậy rộng khoảng ±8 điểm, nên chênh lệch nhỏ hơn
  mức đó **không chứng minh** cấu hình nào tốt hơn.
- Kết quả đã đo: BM25 Hit@5 = **85.3% [77.3, 93.3]**, MRR 0.707.

---

## Q10. Đánh giá chất lượng câu trả lời (generation) thế nào? Có tin được LLM judge không?

**Muốn kiểm tra:** hiểu hạn chế của LLM-as-judge.

**Gợi ý trả lời (`backend/eval/judge.py`):**
- Một lời gọi structured output cho mỗi câu trả lời. Judge **tách câu trả lời thành các claim nguyên
  tử**, và với mỗi claim xác định: được excerpt hỗ trợ không, trích dẫn `[n]` nào, trích dẫn đó có đúng
  không.
  - **Faithfulness** = số claim được hỗ trợ / tổng claim.
  - **Citation precision** = số claim có trích dẫn đúng / số claim có trích dẫn.
  - **Correctness** so với đáp án tham chiếu: 1 / 0.5 / 0.
- Đo thêm: tỉ lệ **từ chối nhầm** (câu in-scope mà trả "not found"), xử lý đúng câu ngoài phạm vi và
  câu vi phạm liêm chính, latency p50/p95, chi phí theo stage.
- **Độ tin cậy:** LLM judge có thiên kiến (thích câu dài, thích văn phong của chính nó). Quy trình đề
  xuất: lấy khoảng 40 câu, gán nhãn tay (không biết cấu hình nào sinh ra), báo **Cohen's κ** cạnh điểm
  judge. Verdict từng claim được lưu trong JSONL để làm việc này.
- v1 dùng heuristic trùng từ nhưng gắn nhãn "RAGAS-compatible", và số trong README v1 đến từ generator
  `--dry-run` giả. Tôi đã **gỡ toàn bộ số đó**. Số answer quality chỉ được công bố sau khi chạy thật
  với API key.

---

## Q11. Làm sao test và chạy CI cho hệ thống phụ thuộc OpenAI mà không tốn tiền, không cần key?

**Muốn kiểm tra:** kỹ năng test hệ thống LLM.

**Gợi ý trả lời:**
- `backend/app/infrastructure/offline.py` có các **stand-in tất định**: `HashEmbeddings` (bag-of-words
  feature hashing, chuẩn hoá L2), chat và utility model heuristic. Chúng chạy **qua đúng pipeline thật**
  (router, cache, retriever, reranker), không mock từng hàm.
- 42 test backend chạy offline: các nhánh refuse và moderation, viết lại câu nối tiếp, cache, PDF có số
  trang, giới hạn upload (413), CORS không wildcard, chi phí phủ mọi lời gọi model, stream lỗi trả thông
  báo chung chung.
- CI (`.github/workflows/ci.yml`): ruff → tải corpus (có checksum) → pytest → **quality gate**
  `eval.run retrieval --offline --configs bm25 --min-hit-rate 0.80` (BM25 không gọi model nên đây là
  **số đo thật**) → smoke test harness → upload report. Frontend: type-check, lint, build.
- Báo cáo offline được **gắn nhãn "smoke test"** rõ ràng, để không ai nhầm số của model giả là kết quả.

---

## Q12. Kiến trúc backend: tại sao Clean Architecture, tại sao SQLite?

**Muốn kiểm tra:** tư duy kiến trúc, biết khi nào đủ.

**Gợi ý trả lời:**
- Các tầng: `api/` (HTTP mỏng) → `services/` (nghiệp vụ) → `domain/` (model và **`Protocol`
  interface**: `VectorStore`, `LLMClient`, `EmbeddingClient`, repository) ← `infrastructure/` (OpenAI,
  Chroma, SQLite). Đồ thị phụ thuộc được dựng ở `bootstrap.py`.
- Lợi ích thực tế: đổi persistence từ in-memory sang SQLite chỉ cần một file implementation, service
  không đổi. Test inject client offline cũng nhờ các interface này.
- **SQLite:** dữ liệu nhỏ (hội thoại, tin nhắn, tài liệu, usage). Dùng stdlib `sqlite3` thay vì ORM để
  latency ổn định trên đường stream nóng. **Kết nối riêng cho mỗi thread** (`threading.local`), **WAL**
  để đọc đồng thời khi đang ghi, `PRAGMA foreign_keys = ON` để `ON DELETE CASCADE` hoạt động.
- Khi lớn hơn thì viết adapter Postgres sau cùng một `Protocol`.

**Nối tiếp:** "Chạy nhiều worker uvicorn thì sao?" Đây là hạn chế cần nói thẳng: BM25 index và semantic
cache nằm **trong bộ nhớ từng process**, nên các worker sẽ không đồng bộ (cache bị xoá ở worker này
nhưng không ở worker khác). Muốn scale ngang cần Redis cho cache, search engine dùng chung (OpenSearch)
hoặc cơ chế pub/sub để rebuild, và Postgres thay SQLite.

---

## Q13. Các guardrail an toàn và bảo mật gồm những gì?

**Muốn kiểm tra:** ý thức an toàn khi đưa LLM ra người dùng.

**Gợi ý trả lời:**
- **Moderation:** OpenAI `omni-moderation-latest` kiểm tra mọi tin nhắn trước khi xử lý (có thể tắt
  trong settings).
- **Router "refuse":** từ chối yêu cầu đạo văn hoặc lách công cụ phát hiện AI, kèm gợi ý giúp theo cách
  hợp lệ (dàn ý, tóm tắt). System prompt cũng nhắc lại quy tắc này.
- **Chống bịa:** prompt "citation-first"; không có căn cứ thì trả câu "not found" cố định; cấm dùng kiến
  thức nền.
- **Ứng dụng web:** CORS dùng danh sách origin cấu hình được (v1 dùng `*` cùng credentials, là cấu hình
  không hợp lệ và không an toàn); giới hạn upload `MAX_UPLOAD_MB` trả 413; kiểm tra đuôi file; lỗi stream
  không lộ exception.
- **Router lỗi thì mặc định `retrieve`,** để sự cố hạ tầng không âm thầm nuốt câu hỏi của sinh viên.

**Nối tiếp:** "Prompt injection trong tài liệu upload thì sao?" Hiện chưa có phòng thủ riêng. Có thể
tách rõ excerpt bằng delimiter, dặn model coi excerpt là dữ liệu chứ không phải lệnh, và đánh giá bằng
một tập tấn công. Dự án chưa có per-user auth và cô lập tài liệu theo người dùng (nằm trong roadmap).

---

## Q14. Bạn tối ưu chi phí và độ trễ LLM thế nào?

**Muốn kiểm tra:** tư duy production và chi phí.

**Gợi ý trả lời:**
- **Model routing theo nhiệm vụ:** `gpt-4.1` chỉ cho bước sinh câu trả lời. Router, rewrite, expansion,
  rerank dùng `gpt-4.1-mini` (rẻ khoảng 5 lần theo bảng giá trong `pricing.py`).
- **Prompt caching của OpenAI:** system prompt **tĩnh và đặt đầu tiên**, nên phần prefix được cache tự
  động. Cached token được theo dõi và tính giá riêng.
- **Router:** câu chào hỏi đi nhánh chitchat bằng model nhỏ, không retrieval.
- **Semantic cache:** cache hit tốn khoảng $0 (chỉ một lời gọi embedding), độ trễ khoảng 50 ms.
- **Bỏ lời gọi thừa:** reranker bỏ qua khi ứng viên ≤ top_n; câu hỏi lượt đầu không cần rewrite; mỗi
  lượt chỉ giữ 20 tin nhắn lịch sử gần nhất.
- **Đo trước khi tối ưu:** `cost_breakdown` theo stage cho biết tiền thực sự đi đâu. Đây cũng là lý do
  phải sửa bộ đo chi phí trước.
- **UX:** SSE stream token và trạng thái từng bước, nên độ trễ cảm nhận thấp hơn độ trễ thực.

---

## Q15. Hạn chế lớn nhất? Nếu có thêm thời gian bạn làm gì?

**Muốn kiểm tra:** tự nhận thức, định hướng.

**Gợi ý trả lời:**
1. **Chưa có ablation đầy đủ với API thật.** Mới có BM25. Cần chạy dense, hybrid, +multi-query, +rerank,
   full để biết mỗi kỹ thuật **có thực sự giúp** không, kèm Cohen's κ giữa judge và người.
2. **Golden set nhỏ, do tác giả viết, tiếng Anh.** Cần câu hỏi thật của sinh viên, và tài liệu tiếng
   Việt (BM25 hiện tách token bằng `\w+`, nên với tiếng Việt cần tách từ, ví dụ pyvi; embedding cần hỗ
   trợ đa ngôn ngữ).
3. **Evidence matching là so khớp từ vựng:** chunk diễn đạt lại ý mà không chứa trích dẫn bị tính là
   không liên quan, tức đánh giá thấp retrieval một chút.
4. **Scale:** BM25 và cache trong bộ nhớ từng process, chunk id theo vị trí (xem Q8 và Q12).
5. **Tính năng:** parent-document retrieval (chunk nhỏ để tìm, chunk lớn để đưa vào context), semantic
   chunking, auth theo người dùng, adapter Postgres.

---

## Câu hỏi nhanh nên thuộc

| Câu hỏi | Trả lời ngắn |
|---|---|
| Chunk size / overlap? | 800 / 150 ký tự |
| k, fetch_k, top_n? | 5, 20, 5 |
| Trọng số hybrid? | vector 0.6, BM25 0.4, `rrf_k = 60` |
| Ngưỡng cache? | cosine ≥ 0.95, chỉnh được trực tiếp |
| Vector store? | Chroma, `hnsw:space = cosine` |
| Sáu mode trả lời? | Answer, Explain, Summarize, Compare, Outline, Quiz |
| Định dạng output có cấu trúc? | `response_format = json_schema` (Pydantic) cho expansion, rerank, judge |
| Observability? | Usage tracker theo stage, endpoint `/api/v1/usage/summary`, LangSmith (tuỳ chọn) |
