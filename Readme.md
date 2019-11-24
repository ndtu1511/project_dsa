Download https://pjreddie.com/media/files/yolov3.weights vào folder `modelserver` sau đó


Chạy lệnh 
```
docker-compose up
```
để chạy server, server sẽ chạy trên http://localhost, chạy url http://localhost/predict để chạy service detect ảnh, ví dụ ở file `client/upload-image.html`