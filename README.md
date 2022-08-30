# Document Scanner Alignment

## Running on Flask
![](https://github.com/wahyu-adi-n/document-alignment-scanner/blob/main/input.png)
![](https://github.com/wahyu-adi-n/document-alignment-scanner/blob/main/output.png)

### Build Image on Docker

```
$ docker build -t wahyuadinugroho/document-alignment:latest .
```

### Pull Image from Dockerhub

```
$ docker pull wahyuadinugroho/focument-alignment:latest
```

### Run Image

```
$ docker run -it --rm -p 5000:5000 wahyuadinugroho/focument-alignment:latest
```

## Running on Streamlit
![](https://github.com/wahyu-adi-n/document-alignment-scanner/blob/main/streamlit.png)

### Clone this repository
```
$ git clone https://github.com/wahyu-adi-n/document-alignment-scanner.git
```

### Move and Change Directory
```
$ cd document-alignment-scanner 
```

### Run on Streamlit
```
$ docker run -it --rm -p 8501:8501 doc_alignment:streamlit
```
