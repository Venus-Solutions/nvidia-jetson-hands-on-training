const express = require('express')
const app = express();
const http = require('http');
const server = http.createServer(app);
const io = require('socket.io')(server);
const port = 8000;

app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

io.on('connection', socket => {
  console.log('connect');
    socket.on("sensor", (data) => {
      console.log(data);
      temp = data.temp;
      photo = data.photo;
      vr = data.vr;

      socket.broadcast.emit('sensorChart', {
        temp, photo, vr
      });
    });
});

server.listen(port, () => {
  console.log(`Jetson Nano app listening on port ${port}!`)
});
