# Library-pytorch-core
Mục lục
Giới thiệu PyTorch & Tensors (Tensor_Pytorch.ipynb)
- Autograd (Autograd.ipynb)
- Lan truyền ngược (Backpropagation.ipynb)
- Hồi quy tuyến tính (Linear_Regression.ipynb)
- Hồi quy Logistic (Logistic_regression.ipynb)
- Mạng truyền thẳng (feedforward_neural_network.ipynb)
- Giảm độ dốc với Autograd (Gradient_Descent_using_autograd.ipynb)
- Mẫu (template.ipynb)
  
1. Giới thiệu PyTorch & Tensors (Tensor_Pytorch.ipynb)
Notebook này cung cấp giới thiệu về PyTorch, nêu bật các tính năng và ưu điểm chính của nó, như API linh hoạt, tích hợp Python và đồ thị tính toán động. Sau đó, nó đi sâu vào cấu trúc dữ liệu cốt lõi của PyTorch: Tensors.

Một số đặc trưng của Pytorch:
- Easy Interface: Pytorch cung cấp API để sử dụng - vì vậy việc triển khải và thực thi code với framework trở nên dễ dàng hơn.
- Python Usage: Bên cạnh việc hỗ trợ mạnh mẽ cho AI, DS thì pytorch có thể tận dụng tất cả các dịch vụ, chức năng và môi trường được cung cấp bởi Python.
- Computational Graph: Pytorch cung cấp các đồ thị tính toán động (dynamic computational graphs) - do đó người dùng có thể thay đổi chúng trong thời gian thực thi.
  
* Một số tiện ích của Pytorch:

Giới thiệu Tensor: Tensor là một kiểu dữ liệu cho phép lưu trữ dữ liệu với số chiều tùy ý, dữ liệu lưu trữ này có thể là một giá trị vô hướng, vector, mảng 1 chiều, mảng 2 chiều hay mảng n chiều.
- Ví dụ về tạo tensor bằng cách sử dụng torch.empty, torch.zeros, torch.ones và torch.rand.
- Kiểm tra kích thước tensor (.size()) và kiểu dữ liệu (.dtype).
- Xây dựng tensor từ dữ liệu hiện có.
- Đối số requires_grad: Tham số này cho biết tensor cần được tính đạo hàm - đây là một bước quan trọng trong quá trình training model.
- Một số phép toán trong tensor: Minh họa phép cộng, trừ, nhân và chia theo từng phần tử bằng cách sử dụng cả toán tử và hàm torch.
- Slicing: Truy cập các phần của tensor bằng cách cắt (ví dụ: x[:, 0], x[1, :], x[1, 1]).
- Reshape (.view()): Thay đổi hình dạng tensor, bao gồm việc sử dụng -1 để tự động suy ra các chiều.
- Khả năng tương tác giữa PyTorch và NumPy:
- Chuyển đổi tensor PyTorch sang mảng NumPy (.numpy()).
- Chuyển đổi mảng NumPy sang tensor PyTorch (torch.from_numpy()).
  Lưu ý quan trọng: Nếu tensor đang được chạy trên CPU, cả 2 object đều chia sẻ cùng một memory location, vì vậy khi thay đổi một object thì object còn lại cũng sẽ được thay đổi.
Kiểm tra và cấu hình thiết bị (Check and config device): Kiểm tra khả năng có CUDA và chuyển tensor sang GPU (.to(device)).
Lưu ý quan trọng: NumPy không chạy được trên GPU vì vậy muốn chuyển torch sang numpy thì phải set device lại thành cpu rồi mới thực hiện bước chuyển.

2. Autograd (Autograd.ipynb)
   
Gói Autograd: Autograd package cung cấp một sự khác biệt tự động cho tất cả các hoạt động của tensor. Rất dễ dàng để sử dụng đạo hàm trong pytorch bằng cách chỉ cho nó biết rằng tensor cần được đạo hàm bằng requires_grad. Với việc thiết lập thuộc tính này, các phép toán trên tensor đều được theo dõi trên một đồ thị tính toán.
- grad_fn: Khi một tensor được tạo ra bởi kết quả của một phép tính, nó sẽ tạo một thuộc tính grad_fn, tham chiếu đến một hàm đã tạo tensor.
- Tính đạo hàm với lan truyền ngược: Khi hoàn tất quá trình tính toán, ta có thể gọi .backward() và tất cả giá trị đạo hàm sẽ được tính toán một cách tự động. Giá trị đạo hàm của những tensor này sẽ được tích lũy vào trong thuộc tính .grad. Nó chính là đạo hàm riêng của tensor.
- Dừng một tensor không theo dõi lịch sử: Trong quá trình huấn luyện, khi chúng ta muốn cập nhật trọng số thì thao tác cập nhật này không nên là một phần của phép tính đạo hàm. Chúng ta có 3 sự lựa chọn cho việc dừng quá trình đạo hàm và cập nhật tham số như sau:
  - x.requires_grad_(False): thay đổi yêu cầu ngay tại vị trí cần yêu cầu đạo hàm.
  - x.detach(): Lấy một tensor mới với nội dung tương tự nhưng không yêu cầu tính đạo hàm.
  - wrap in with torch.no_grad().
Gradients rỗng (.zero_()): Với backward() ta sẽ có đạo hàm tích lũy bên trong thuộc tính .grad. Chúng ta cần cẩn thận với nó trong quá trình tối ưu. zero_() sẽ tránh lưu lại kết quả của lần đạo hàm trước đó.

3. Lan truyền ngược (Backpropagation.ipynb)
   
- Dự đoán và tính toán Loss: Định nghĩa một mô hình tuyến tính đơn giản $y\_{predict} = x \* w$ và hàm mất mát bình phương trung bình L=(y_predict−y) 
- Backward pass (lan truyền ngược): Gọi loss.backward() để tính toán gradient của hàm mất mát theo trọng số w.-
- Cập nhật trọng số: Cập nhật trọng số w bằng cách sử dụng quy tắc giảm độ dốc đơn giản: w -= 0.001 * w.grad. Việc cập nhật được thực hiện trong with torch.no_grad(): để ngăn không cho nó trở thành một phần của đồ thị tính toán.
- Đặt lại gradient về 0: Đặt gradient của w về 0 bằng cách sử dụng w.grad.zero_() sau mỗi lần cập nhật.
  
4. Hồi quy tuyến tính (Linear_Regression.ipynb)
   
- Siêu tham số: Xác định input_size, output_size, num_epochs và learning_rate.
- Tạo tập dữ liệu: Tạo tập dữ liệu tổng hợp x_train và y_train bằng cách sử dụng mảng NumPy.
- Mô hình tuyến tính: Định nghĩa mô hình hồi quy tuyến tính bằng cách sử dụng nn.Linear, xử lý trọng số và độ lệch nội bộ.
- Hàm mất mát và Bộ tối ưu hóa: Sử dụng nn.MSELoss() làm tiêu chí cho lỗi bình phương trung bình và torch.optim.SGD() (Giảm độ dốc ngẫu nhiên) làm bộ tối ưu hóa.
* Huấn luyện mô hình:
- Chuyển đổi mảng NumPy sang tensor PyTorch cho đầu vào và mục tiêu.
- Thực hiện truyền xuôi để nhận dự đoán.
- Tính toán hàm mất mát.
- Thực hiện truyền ngược để tính toán gradient (loss.backward()).
- Cập nhật các tham số mô hình bằng bộ tối ưu hóa (optimizer.step()).
- Đặt lại gradient về 0 (optimizer.zero_grad()).
* Dự đoán và trực quan hóa:
- Thực hiện dự đoán trên dữ liệu huấn luyện bằng mô hình đã huấn luyện.
- Chuyển đổi dự đoán trở lại NumPy để vẽ biểu đồ.
- Trực quan hóa các điểm dữ liệu ban đầu và đường hồi quy đã khớp bằng matplotlib.
Lưu mô hình: Lưu từ điển trạng thái của mô hình đã huấn luyện vào một tệp (linear_model.ckpt).

5. Hồi quy Logistic (Logistic_regression.ipynb)

* Tập dữ liệu MNIST:
- Tải xuống và tải tập dữ liệu MNIST bằng cách sử dụng torchvision.datasets.MNIST.
- Áp dụng transforms.ToTensor() để chuyển đổi hình ảnh thành tensor PyTorch.
- Tạo các thể hiện DataLoader để huấn luyện và kiểm tra, cho phép xử lý theo lô nhỏ và xáo trộn.
- Mô hình hồi quy Logistic: Định nghĩa mô hình là một lớp nn.Linear đơn giản ánh xạ hình ảnh đầu vào được làm phẳng thành đầu ra num_classes.
- Hàm mất mát và Bộ tối ưu hóa: Sử dụng nn.CrossEntropyLoss() cho phân loại đa lớp và torch.optim.SGD() làm bộ tối ưu hóa.
* Huấn luyện mô hình:
- Lặp qua các kỷ nguyên và các lô nhỏ từ train_loader.
- Định hình lại hình ảnh thành (batch_size, input_size).
- Thực hiện truyền xuôi để nhận đầu ra của mô hình.
- Tính toán hàm mất mát bằng tiêu chí đã định nghĩa.
- Thực hiện truyền ngược và tối ưu hóa trọng số.
- In mất mát huấn luyện định kỳ.

* Kiểm tra mô hình:
- Đánh giá mô hình trên tập dữ liệu thử nghiệm bằng cách sử dụng torch.no_grad() để tắt tính toán gradient trong quá trình đánh giá.
- Tính toán độ chính xác bằng cách so sánh nhãn dự đoán với nhãn thực tế.
Lưu mô hình: Lưu từ điển trạng thái của mô hình đã huấn luyện vào một tệp (logistic_regression_model.ckpt).

6. Mạng truyền thẳng (feedforward_neural_network.ipynb)

- Cấu hình thiết bị: Kiểm tra khả năng có CUDA và đặt thiết bị thành GPU nếu có, nếu không thì CPU.
- Siêu tham số: Xác định input_size, hidden_size, num_classes, num_epochs, batch_size và learning_rate.
- Tập dữ liệu MNIST: Tải tập dữ liệu MNIST và tạo các thể hiện DataLoader, tương tự như ví dụ hồi quy logistic.
* Kiến trúc mạng nơ-ron:
- Định nghĩa lớp NeuralNet kế thừa từ nn.Module.
- Triển khai lớp được kết nối đầy đủ (fc1), hàm kích hoạt ReLU và một lớp được kết nối đầy đủ khác (fc2).
- Định nghĩa quá trình truyền xuôi (forward), chỉ định cách dữ liệu chảy qua mạng.
- Tạo mô hình: Khởi tạo mô hình NeuralNet và chuyển nó đến thiết bị đã cấu hình.
- Hàm mất mát và Bộ tối ưu hóa: Sử dụng nn.CrossEntropyLoss() và torch.optim.SGD().
* Huấn luyện mô hình:
- Vòng lặp huấn luyện tương tự như hồi quy logistic, bao gồm định hình lại hình ảnh, chuyển chúng đến thiết bị, truyền xuôi và truyền ngược, và tối ưu hóa.
- In mất mát huấn luyện định kỳ.
- Kiểm tra mô hình: Đánh giá độ chính xác của mô hình trên tập kiểm tra, tương tự như ví dụ hồi quy logistic.
Lưu mô hình: Lưu từ điển trạng thái của mô hình đã huấn luyện.

7. Giảm độ dốc với Autograd (Gradient_Descent_using_autograd.ipynb)
   
- Khởi tạo: Định nghĩa đầu vào x, mục tiêu y và khởi tạo trọng số w với requires_grad=True.
- Hàm truyền xuôi: Định nghĩa quá trình truyền xuôi forward(x) = x * w.
- Hàm mất mát: Định nghĩa hàm mất mát bình phương trung bình (MSE).
  
* Vòng lặp huấn luyện:
- Đặt learning_rate và epochs.
- Trong mỗi kỷ nguyên:
- Thực hiện truyền xuôi để nhận dự đoán y_pred.
- Tính toán hàm mất mát l.
- Thực hiện truyền ngược (l.backward()) để tính toán gradient của hàm mất mát theo w.
- Cập nhật trọng số w bằng cách sử dụng gradient đã tính toán và learning_rate trong with torch.no_grad(): để ngăn không cho thao tác này trở thành một phần của đồ thị.
- Đặt lại w.grad về 0 bằng w.grad.zero_() cho lần lặp tiếp theo.
- In trọng số và mất mát hiện tại định kỳ.
* Dự đoán: Sau khi huấn luyện, thực hiện dự đoán cho một giá trị x mới bằng cách sử dụng w đã tối ưu hóa.
  
9. Mẫu (template.ipynb)

* Ví dụ Autograd:
- Minh họa tính toán gradient cho một phương trình đơn giản y = w*x + b, hiển thị x.grad, w.grad và b.grad.
- Minh họa autograd với một lớp được kết nối đầy đủ, tính toán dL/dW và dL/dB và hiển thị hiệu ứng của bước tối ưu hóa trên hàm mất mát.
- Tải dữ liệu Numpy: Cho biết cách chuyển đổi mảng NumPy sang tensor PyTorch (torch.from_numpy()) và ngược lại (.numpy()).
- Input pipeline (với torchvision.datasets):
- Minh họa cách tải tập dữ liệu (trong trường hợp này là CIFAR10) bằng cách sử dụng torchvision.datasets.
- Giải thích cách sử dụng DataLoader để xử lý theo lô nhỏ và xáo trộn.
- Cung cấp ví dụ về cách lặp qua DataLoader.
- Input pipeline cho tập dữ liệu tùy chỉnh: Cung cấp một mẫu để tạo lớp tập dữ liệu tùy chỉnh bằng cách kế thừa từ torch.utils.data.Dataset, phác thảo các phương thức __init__, __getitem__ và __len__.
* Các mô hình được đào tạo trước:
- Cho biết cách tải một mô hình được đào tạo trước (ví dụ: ResNet-18) từ torchvision.models.
- Minh họa việc tinh chỉnh bằng cách đóng băng các lớp trước đó (param.requires_grad = False) và thay thế lớp phân loại cuối cùng.

* Lưu và tải mô hình:
- Giải thích cách lưu và tải toàn bộ mô hình.
- Đề xuất và minh họa việc lưu và tải chỉ từ điển trạng thái của mô hình (tham số) để linh hoạt hơn.
