# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (T4)

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy? 
- Record này mô tả toàn bộ bức ảnh, không mô tả một vật thể riêng lẻ, vì model classification chỉ chọn một lớp cho cả ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách class đi kèm taxonomy và checkpoint của model, không phải do model tự nghĩ ra trong lúc dự đoán.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Cần ghi cả `class_id`, `class_name` và `taxonomy_name` để biết chính xác lớp nào được chọn và lớp đó thuộc bộ class nào. Cùng một tên lớp có thể có ý nghĩa khác nhau trong các taxonomy khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần quy định chủ đề chính của ảnh, thứ tự ưu tiên giữa các vật thể và cách xử lý khi ảnh có nhiều vật hoặc nhiều chủ đề.
6. Vì sao không được lấy `score` của model làm ground truth?
- Không được lấy model score làm ground truth vì score chỉ thể hiện mức độ tin tưởng của model. Ground truth phải do annotator tạo theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
- `class_name` = oven;
   `score` = 0.686791;
   `bbox_xyxy` = [0.11, 187.54, 195.96, 292.81];
   `bbox_width` = 195.85;
   `bbox_height` = 105.26.
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Bounding box có góc trên-trái tại tọa độ `(0.11, 187.54)` và góc
dưới-phải tại tọa độ `(195.96, 292.81)`. Box rộng `195.85` pixel và
cao `105.26` pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- chọn hai ngưỡng `0.20` và `0.60`
-  Ở threshold `0.20` có `17` prediction.
   Ở threshold `0.60` có `6` prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ threshold, model giữ lại nhiều prediction hơn. Reviewer phải kiểm tra thêm các hộp sai, hộp trùng nhau, class sai và các vật thể có score thấp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Bounding box phải bao phủ toàn bộ phần nhìn thấy của vật thể, không bao gồm quá nhiều vùng nền và không được bỏ sót phần vật thể nhìn thấy.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Trên ảnh `kitchen`, vật thể `cup` nằm trên bàn và bị che/chồng lấn một phần bởi các vật thể hoặc bounding box khác. Tôi chọn escalation với Lab Coach vì chưa chắc phần bị che phải được vẽ bounding box như thế nào. Hỏi Lab Coach giúp các annotator xử lý trường hợp này thống nhất thay vì tự suy đoán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
-  `instance_id` = `kitchen-010`;
   `class_name` = `bowl`;
   `score` = `0.372406`;
   `polygon_point_count` = `12`;
   `polygon_xy`: 
   [
      [
        155.0,
        170.0
      ],
      [
        155.0,
        177.0
      ],
      [
        156.0,
        178.0
      ],
      [
        156.0,
        179.0
      ]
   ]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- So với bounding box, polygon cung cấp thêm thông tin chi tiết về hình dạng và đường biên thật của vật thể. Bounding box chỉ là hình chữ nhật bao quanh vật thể, còn polygon bám sát các cạnh, góc, phần cong và vùng thực sự thuộc về vật thể. Vì vậy polygon phù hợp hơn khi cần biết chính xác từng pixel hoặc hình dạng của vật thể.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id` dùng để phân biệt từng vật thể riêng biệt trong ảnh, ngay cả khi các vật thể đó có cùng `class_name`.

Ví dụ, hai cái bowl đều có thể thuộc class `bowl`, nhưng sẽ có hai `instance_id` khác nhau như `kitchen-010` và `kitchen-011`.

- `instance_id` không phải `class_id` vì `class_id` đại diện cho loại vật thể. Nó cũng không phải tracking ID vì nó chỉ dùng để định danh instance trong output của bài lab, không theo dõi vật thể qua nhiều khung hình.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Quy tắc vẽ biên mask: polygon phải bám sát đường biên nhìn thấy của vật thể, chỉ bao phủ phần thuộc về vật thể và không bao gồm vùng nền. Nếu vật thể bị che khuất, annotator vẽ theo phần nhìn thấy và thực hiện escalation nếu guideline chưa quy định rõ cách xử lý.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Trên ảnh `kitchen`, các vật thể như bowl nằm gần hoặc chồng lên nhau nên một số đường biên có thể không rõ. Tôi chọn escalation với Lab Coach nếu guideline chưa quy định cụ thể, để xác định phần biên cần vẽ và đảm bảo các annotator xử lý giống nhau.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):


| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một class cho toàn bộ ảnh | Ảnh có nhiều vật hoặc nhiều chủ đề; model chỉ chọn một class | Chọn class theo chủ đề chính và guideline | Kiểm tra class có phù hợp với ảnh và guideline không |
| Phát hiện vật thể | Một class và một bounding box cho mỗi vật thể | Vật thể bị che, box chồng nhau hoặc prediction thay đổi theo threshold | Vẽ box bao phủ phần nhìn thấy của vật thể | Kiểm tra class, vị trí box, box sai và vật bị bỏ sót |
| Instance segmentation | Một class và một polygon cho mỗi instance | Biên mờ, vật thể chạm nhau hoặc bị che | Vẽ polygon bám theo đường biên vật thể | Kiểm tra polygon có bao phủ đúng vật thể và không lấy nền không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ sử dụng ba ảnh COCO công khai được cung cấp trong notebook, không đưa ảnh cá nhân, dữ liệu nội bộ hoặc dữ liệu nhạy cảm vào bài lab.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý và báo cho Lab Coach hoặc giảng viên phụ trách.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
