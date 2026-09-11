# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 9/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  [`468`, `cab`, `1`, `0.5109915`, `ImageNet-1k`]

- Record này mô tả toàn ảnh như thế nào?
  Mô hình này nhận diện các vật thể trong ảnh, và tự phân loại. Một vật thể trong bức ảnh được phân theo
  ['sample_id', 'coco_image_id', 'image_width', 'image_height', 'task', 'taxonomy_name', 'model_file','model_sha256', 'ultralytics_version','rank','class_id', 'class_name', 'score']

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Các class list được tạo ra thủ công từ nhà phát triển Model Yolo "YOLO11n-cls trả về danh sách các lớp ImageNet-1K được xếp hạng cho toàn bộ ảnh.", model biết được 1000 vật thể

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  Vì các nhãn có ID với tên lớp riêng đều nằm trong danh sách trong tập nhãn taxonomy ImageNet-1k

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Số lượng đối tượng trong ảnh,
  Mỗi đối tượng được gắn nhãn theo lớp nào, chọn đối tượng chính và phụ nếu xảy ra chồng chéo
  Hướng xử lý các đối tượng trong ảnh nếu
  - Ảnh nhỏ hoặc xa
  - Ảnh bị che
  - Ảnh ở ngoài
  - Nhiều vật thể trong ảnh

- Vì sao model score không phải ground truth?
  Vì model score là kết quả do máy/mô hình dự đoán không phải là kết quả cuối cùng của người Annotator quyết định.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  [`bus`,`0.912557`,`[93.17,187.95,223.01,320.91]`, `129.84`, `132.96`]

- Diễn giải vị trí box bằng lời:  
  Box là khung xác định vật thể, xác định vị trí và phạm vi từng vật thể. Mỗi bức ảnh có nhiều vật thể có nhiều box riêng cho từng vật thể

- So sánh số prediction ở hai threshold:
  Ở threshold là 0.20 có 17 vật thể được nhận ra, còn với 0.35 thì giảm còn 11 vật thể

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  Khi threshold thấp, mô hình giữ lại nhiều prediction hơn nên độ bao phủ (recall) thường tăng, nhưng reviewer phải kiểm tra nhiều box hơn và số false positive cũng có thể tăng. Khi threshold cao, số prediction giảm nên reviewer ít việc hơn, nhưng có nguy cơ bỏ sót vật thể thật.

- Đề xuất một quy tắc box chặt:
  Box nên bao sát phần nhìn thấy của vật thể, hạn chế vùng nền thừa, đồng thời không cắt mất phần vật thể đang nhìn thấy.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  Guideline cần quy định rõ box theo phần nhìn thấy hay ước lượng toàn bộ vật thể, và mức độ che khuất tối thiểu/tối đa được chấp nhận. Những trường hợp không xác định được loại vật thể, bị che quá nhiều, hoặc có nhiều cách đặt box hợp lý nên được escalate thống nhất cách xử lý.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  [`traffic-001`, `car`, `0.891092`, `92`, "polygon_xy": [[497.0,287.0],[497.0,287.0],[492.0,287.0],]]
- Polygon bổ sung chi tiết gì so với box?
  Polygon đa giác cố gắng xác định điêm vị trí phân biệt dựa theo biên của vật thể cố gắng xác định vẽ đa giác gần vật thể. Khác với box chỉ hình vuông/nhật vật thể

- `instance_id` dùng để làm gì và không phải loại ID nào?
  `instance_id` dùng để xác định, đánh số từng vật thể trong ảnh, máy nhận diện được bao nhiêu vật thể thì sẽ có instance_id tương ứng. instance_id không phải là class_id

- Đề xuất một quy tắc biên mask:
  Mask nên bám sát đường biên thực tế nhìn thấy của vật thể, không lấy dư nhiều vùng nền và không cắt mất phần vật thể nhìn thấy được. Nên áp dụng cùng một tiêu chuẩn cho toàn bộ dataset để đảm bảo tính nhất quán.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  Khi biên mask được đặt ở đâu khi ranh giới không rõ, hai vật thể chạm nhau thì tách mask theo nguyên tắc nào, và vật thể bị che thì chỉ vẽ phần nhìn thấy hay ước lượng cả phần bị che.
  Nếu annotator không thể xác định chắc chắn theo guideline hiện có, trường hợp đó nên escalate cho reviewer \để đưa ra cách xử lý thống nhất cho toàn bộ dataset.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Đơn vị/định dạng ground truth  | Lỗi hoặc điểm mơ hồ quan sát được                                                                         | Annotator làm gì?                                                                              | Reviewer xem gì?                                                                                                    |
| --------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | 1 nhãn hoặc class cho mỗi ảnh  | Ảnh mờ, nhiều đối tượng, class dễ nhầm, đối tượng không thuộc danh sách class                             | Chọn class đúng theo guideline; đánh dấu/escalate nếu không chắc chắn                          | Kiểm tra class có đúng với nội dung ảnh và guideline không; kiểm tra các case mơ hồ                                 |
| Phát hiện vật thể     | 1 bounding box cho mỗi vật thể | Box quá rộng/hẹp, bỏ sót object, box trùng, sai class; object bị che/cắt mép                              | Vẽ box sát phần vật thể theo quy tắc; gán class; xử lý che khuất/cắt mép theo guideline        | Kiểm tra đủ object chưa; vị trí/kích thước box; class; box trùng; tính nhất quán                                    |
| Instance segmentation | 1 mask/polygon cho mỗi vật thể | Biên mask không chính xác, mask dính/chồng nhau, bỏ sót vùng; khó xác định biên khi mờ/tiếp xúc/che khuất | vẽ mask sát biên vật thể; tách từng instance; gán class và instance_id; escalate case không rõ | Kiểm tra độ chính xác biên mask; từng instance có được tách đúng không; class/ID; vùng thiếu/thừa và tính nhất quán |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Không sao chép, tải xuống, chia sẻ hoặc sử dụng dữ liệu dự án ngoài phạm vi công việc và công cụ được phép.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  Reviewer / Team Lead / người phụ trách dự án để kiểm tra và quyết định cách xử lý.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
