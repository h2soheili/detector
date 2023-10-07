```
 this is fastapi app
 start and see docs at localhost:port/docs
    
 for start stream do:
 
 POST http://127.0.0.1:4000/v1/stream   
 wit one of these configs
 
{
  "id": 1,
  "name": "for start webcam",
  "source": "0",
  "boundary": null,
  "img_size": [
    640,
    640
  ],
  "stride": 32,
  "auto": true,
  "vid_stride": 1,
  "classes": []
}

{
  "id": 2,
  "name": "online stream",
  "source": "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8",
  "boundary": null,
  "img_size": [
    640,
    640
  ],
  "stride": 32,
  "auto": true,
  "vid_stride": 1,
  "classes": []
}

{
  "id": 3,
  "name": "online stream 2",
  "source": "https://cph-p2p-msl.akamaized.net/hls/live/2000341/test/master.m3u8",
  "boundary": null,
  "img_size": [
    640,
    640
  ],
  "stride": 32,
  "auto": true,
  "vid_stride": 1,
  "classes": []
}

```