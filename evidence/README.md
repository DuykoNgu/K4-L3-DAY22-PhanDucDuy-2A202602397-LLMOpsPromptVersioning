# Phân tích kết quả RAGAS

Cùng 50 cặp câu hỏi/đáp án chuẩn được chạy qua V1 và V2. Mỗi câu trả lời dùng 3 đoạn tài liệu từ FAISS. Mô hình tạo câu trả lời: Ollama `llama3.2:3b`; mô hình chấm RAGAS: `qwen2.5:3b`; embeddings: `nomic-embed-text`.

| Chỉ số | V1 | V2 | Cao hơn |
|---|---:|---:|---|
| faithfulness | 0.7643 | 0.8888 | V2 |
| answer_relevancy | 0.6040 | 0.8741 | V2 |
| context_recall | 0.9631 | 0.9631 | Hòa |
| context_precision | 0.9556 | 0.9483 | V1 |

**Ngưỡng faithfulness ≥ 0,8:** đạt. V2 thay prompt tổng hợp dài bằng câu trả lời ngắn, chỉ nêu dữ kiện có trong context; faithfulness thay đổi +0.1244 so với V1.

V1 có 5 tác vụ RAGAS bị quá thời gian khi Ollama chạy đồng thời nhiều yêu cầu; điểm trung bình bỏ qua giá trị không hợp lệ. Lượt chấm V2 cuối có 1 lỗi phân tích đầu ra trên 200 tác vụ; các điểm trung bình cũng chỉ dùng giá trị hợp lệ. Các tệp JSON và ảnh điểm trong thư mục này lấy từ các lần chạy thật, không phải điểm mẫu.
