const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 5555 }); // Use an available port

wss.on('connection', function connection(ws) {
  console.log('Client connected');

  ws.on('message', function incoming(message) {
    // Convert the message from Buffer to string
    const messageString = message.toString();

    // Parse and debug print the incoming message
    try {
      const data = JSON.parse(messageString);
      console.log('Received message:', data);
    } catch (error) {
      console.log('Received message (not JSON):', messageString);
    }

    // Broadcast the message to all connected clients except the sender
    wss.clients.forEach(function each(client) {
      if (client !== ws && client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  });

  ws.on('close', function close() {
    console.log('Client disconnected');
  });
});
