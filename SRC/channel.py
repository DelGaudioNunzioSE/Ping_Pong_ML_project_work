import socket
import time
import queue
import threading
import select
import struct
from abc import ABC, abstractmethod

#These parameters and constants are critical for managing the behavior of message transmission, reception, and handling within a communication system, ensuring efficient and reliable data exchange.
MESSAGE_NORMAL = 128
MAX_MESSAGE_LENGTH=65535-MESSAGE_NORMAL
MESSAGE_PING   = 1 #This defines the value used for a ping message, typically for checking connectivity.
MESSAGE_REFUSED = 2 #This sets the value indicating that a message was refused.
MESSAGE_HEADER_SIZE=2
MAX_OUTBOUND_QUEUE = 15
MIN_OUTBOUND_BUFFER = 4096
RECV_SIZE=MAX_MESSAGE_LENGTH+MESSAGE_HEADER_SIZE
WAIT_TIME=0.2
PING_TIME=2.0

#A custom exception class that inherits from IOError. It is used to handle errors specific to channel operations.
class ChannelError(IOError):
    pass

#An abstract base class (ABC) for a communication channel. It defines the essential methods that any concrete implementation of a channel must provide.
class AbstractChannel(ABC):
    @abstractmethod
    def send(self, message):
        pass

    @abstractmethod
    def send_refuse(self):
        pass

    @abstractmethod
    def receive(self, timeout=None):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def is_closed(self):
        pass

    @abstractmethod
    def is_refused(self):
        pass

    @abstractmethod
    def is_finished(self):
        pass

    @abstractmethod
    def last_activity_time(self):
        pass


class BaseChannel(AbstractChannel):
    def __init__(self):
        self.lock = threading.RLock() #RLock is used for concurrency management.
        self.sock = None #socket at first is none
        self.inbound_queue = queue.SimpleQueue() #queques for incoming messages
        self.outbound_queue = queue.SimpleQueue() #queques for outcoming messages
        self.inbound_buffer = b'' #buffer for input data.
        self.outbound_buffer = b'' #buffer for output data.
        self.closed = False
        self.last_write_time = time.time()
        self.last_read_time = time.time()
        
        #Various flags to track channel status.
        self.found_error=False
        self.refused = False
        self.reader_finished = False
        self.writer_finished = False

        #Starts two threads: one for reading (reader_thread) and one for writing (writer_thread).
        t1=threading.Thread(target=self.reader_thread)
        t1.start()
        t2=threading.Thread(target=self.writer_thread)
        t2.start()

   
    #If another thread tries to acquire the lock while it has already been acquired, it will have to wait until the lock is released.
    def set_socket(self, sock):
        with self.lock:  #Critical Code Protection: Using with self.lock ensures that the code within the lock is executed atomically, preventing race conditions. 
            try:
                if self.sock:
                    self.sock.close()
            finally:
                self.sock=sock
            self.inbound_buffer = b''
            self.outbound_buffer = b''

    #Sends a message by adding it to the outbound queue, ensuring it does not exceed the maximum length or queue size.
    def send(self, message):
        with self.lock: #The function begins by acquiring the lock to ensure that access to the code within the with self.lock block is thread-safe.
            if self.closed: #If the channel is closed (self.closed is True), the function returns immediately without doing anything.
                return
            if self.refused:
                raise ChannelError('Send to a connection refused by peer') #If the channel has been rejected by the peer (self.refused is True), a ChannelError exception is raised with an appropriate message.
        if len(message)>MAX_MESSAGE_LENGTH: #Message Length Control: If the message to be sent is longer than MAX_MESSAGE_LENGTH, a ChannelError exception is raised indicating that the message is too long.
                raise ChannelError('Message too long in send')
        try:
            #function checks the size of the outbound message queue (self.outbound_queue). If the queue is larger than MAX_OUTBOUND_QUEUE, removes older messages to make room for new messages.
            while self.outbound_queue.qsize()>MAX_OUTBOUND_QUEUE: 
                self.outbound_queue.get(block=False) #The removal occurs in a non-blocking manner (block=False), which means that if the queue is empty, a queue.Empty exception is raised, which is caught and handled safely with pass.
        except queue.Empty:
            pass #null instruction, used to represent an empty block of code.
        self.outbound_queue.put(message)

    #sends a refusal message by adding it to the outbound buffer.
    def send_refuse(self):
        m=struct.pack('!H', MESSAGE_REFUSED) #Use the struct library to create a binary message. !H specifies that the data should be packed as an unsigned short in network byte order (big-endian).
        with self.lock: #Acquires the lock to ensure that the operation of modifying the output buffer (self.outbound_buffer) is thread safe.
            self.outbound_buffer += m 
            

    #Receives a message from the inbound queue, optionally with a timeout.
    def receive(self, timeout=None):
        with self.lock:
            if self.refused:
                raise ChannelError('Receive from a connection refused by peer')
            if self.closed:
                return None

        #The timeout parameter is used to determine how long the thread must wait to receive a message from the queue (inbound_queue) before continuing. 
        #The timeout is important to prevent the thread from being blocked indefinitely waiting for a message, instead allowing you to handle situations where messages may not arrive within a reasonable time.
        try:
            #No Timeout Specified (Non-Blocking): attempts to get a message from the queue in a non-blocking manner. If the queue is empty, it raises a queue.Empty exception which is handled in the except block.
            if not timeout:
                return self.inbound_queue.get(block=False)

            #Timeout Specified and Greater than or Equal to Zero (Timeout Blocking): If a message arrives within the specified time, it is returned; otherwise, if the time expires, a queue.Empty exception is raised
            elif timeout>=0.0:
                return self.inbound_queue.get(timeout=timeout)
            else:
                #Timeout Less Than Zero (Blocking Without Timeout): the thread will wait indefinitely until a message becomes available in the queue.
                return self.inbound_queue.get()
        except queue.Empty:
            return None

    #Closes the channel.
    def close(self):
        with self.lock:
            self.closed=True

    #These methods check the status of the channel, such as if it's closed, refused, finished, or retrieve the last activity time.
    def is_closed(self):
        with self.lock:
            return self.closed

    def is_refused(self):
        with self.lock:
            return self.refused

    def is_finished(self):
        with self.lock:
            return self.reader_finished and self.writer_finished 

    #returns the last time read or write activity occurred on the socket. It is useful for monitoring channel activity and detecting periods of inactivity.
    def last_activity_time(self):
        with self.lock:
            return max(self.last_read_time, self.last_write_time)

    #The reader thread (reader_thread) is responsible for reading data from the socket as it becomes available. It runs in a continuous loop until the channel is closed.
    def reader_thread(self):
        while not self.is_closed():
            fd=None
            with self.lock:
                if self.sock:
                    fd=self.sock.fileno() #Gets the socket descriptor (fd) file, if it exists.
            if fd is None: #If fd is None, the thread waits for a short time (WAIT_TIME) and then starts the loop again.
                time.sleep(WAIT_TIME)
                continue
            rready, _, xready=select.select([fd],[],[fd],WAIT_TIME) #Use select to monitor the socket file descriptor for read (rready) and errors (xready), with a timeout of WAIT_TIME.
            with self.lock: #Reacquires the lock to ensure that the socket has not been modified by other threads.
                if not self.sock or fd!=self.sock.fileno():
                    continue
                if xready: #Checks for errors (xready) and handles them by setting self.found_error to True.
                    self.found_error=True 
                if rready and not self.found_error: #If there is data to read (rready) and there are no errors, call do_read to read the data from the socket.
                    self.do_read()
            self.check_error()
            # End of while
        with self.lock:
            self.reader_finished=True #read thread has finished its execution.
            try:
                if self.writer_finished and self.sock: #If the writing thread has finished and the socket (self.sock) exists, close the socket by calling self.sock.close().
                    self.sock.close()
            except IOError:
                pass

    def writer_thread(self):
        while not self.is_closed():
            self.prepare_write()
            fd=None
            with self.lock:
                if self.sock:
                    fd=self.sock.fileno()
            if fd is None:
                time.sleep(WAIT_TIME)
                continue
            _, wready, xready=select.select([],[fd],[fd],WAIT_TIME)
            with self.lock:
                if not self.sock or fd!=self.sock.fileno():
                    continue
                if xready:
                    self.found_error=True
                if wready and not self.found_error:
                    self.do_write()
            self.check_error()
            # End of while
        with self.lock:
            self.writer_finished=True
            try:
                if self.reader_finished and self.sock:
                    self.sock.close()
            except IOError:
                pass



    #Prepares data to be written to the socket, ensuring the buffer meets the minimum size requirement and handling ping messages. 
    #It ensures that the output buffer is adequately filled before the write attempt and sends periodic pings to keep the connection alive. 
    #It uses thread-safe code blocks to ensure that operations on shared variables are performed without interference between concurrent threads.
    def prepare_write(self):
        with self.lock:
            bl=len(self.outbound_buffer)
        while bl<MIN_OUTBOUND_BUFFER: #Continue to fill the output buffer until its length is less than MIN_OUTBOUND_BUFFER.
            try:
                if bl==0:
                    m=self.outbound_queue.get(timeout=PING_TIME) #attempts to get a message from the outbound queue (self.outbound_queue) with a timeout of PING_TIME
                else:
                    m=self.outbound_queue.get(block=False) #attempts to get a message from the output queue in a non-blocking manner (block=False)
            except queue.Empty:
                break
            em=self.encode_message(m) #obtaining encoded message
            with self.lock:
                self.outbound_buffer += em
                bl=len(self.outbound_buffer)
        if bl==0:
            ping=struct.pack('!H', MESSAGE_PING) #If after the fill cycle the output buffer is still empty (bl == 0), a ping message is created.
            with self.lock:
                self.outbound_buffer+=ping

    #Encodes messages with a header indicating their length and type.
    def encode_message(self, message):
        n=len(message)
        if n>MAX_MESSAGE_LENGTH:
            print('*** Warning: Attempted sending a message too long!')
            return ''
        h=struct.pack('!H', MESSAGE_NORMAL+n) #create a binary header. !H indicates an unsigned short integer in network byte order (big-endian).
        return h+message


    #Writes data from the outbound buffer to the socket.
    def do_write(self):
        if self.outbound_buffer:
            try:
                n=self.sock.send(self.outbound_buffer)#Attempts to send data in the output buffer to the socket
            except IOError:
                n=-1
            if n<1:
                self.found_error=True
            else:
                self.outbound_buffer = self.outbound_buffer[n:] #If sending is successful, removes the sent data from the output buffer
                self.last_write_time=time.time()

    #Reads data from the socket into the inbound buffer.
    def do_read(self):
        data=b'' #Initialize data as an empty byte string.
        try:
            data=self.sock.recv(RECV_SIZE) #Attempts to read data from the socket
        except IOError:
            self.found_error=True
        if data:
            self.inbound_buffer += data
            self.last_read_time=time.time()
            self.parse_messages() #processes messages in the input buffer.
        else:
            self.found_error=True #If no data is read (indicating the connection was closed), set self.found_error to True.

    #Checks for errors and handles closing the socket if necessary.
    def check_error(self):
        with self.lock:
            if self.found_error:
                self.found_error=False
                try:
                    self.sock.close()
                finally:
                    self.sock=None
                self.on_error()


    #Placeholder method for handling errors, intended to be overridden by subclasses.
    def on_error(self):
        pass

    #Posts a message to the inbound queue.
    def post_message(self, message):
        self.inbound_queue.put(message)

    #Parses incoming messages from the inbound buffer and handles them according to their type.
    def parse_messages(self):
        while True:
            blen=len(self.inbound_buffer)
            if blen<MESSAGE_HEADER_SIZE: #If the length is less than the size of the message header, it breaks the loop as there is not enough data for a complete message.
                break
            header=self.inbound_buffer[:MESSAGE_HEADER_SIZE] #Extracts the message header from the input buffer
            msg_code=struct.unpack('!H', header)[0] #decode the header and get the message code. It contains a value that includes information about both the message type and its length.
            msg_length=max(msg_code-MESSAGE_NORMAL, 0) #in case msg_code is less than MESSAGE_NORMAL), max(..., 0) ensures that msg_length is at least 0.
            msg_code=msg_code - msg_length #After calculating msg_length, the code restores the message code value to identify the exact type of message.
            if msg_code==MESSAGE_NORMAL:
                if blen<MESSAGE_HEADER_SIZE+msg_length: #Checks whether there is enough data in the input buffer for the entire message
                    break
                message=self.inbound_buffer[MESSAGE_HEADER_SIZE:
                                            MESSAGE_HEADER_SIZE+msg_length] #Extracts the message from the input buffer
                self.inbound_buffer=self.inbound_buffer[
                                            MESSAGE_HEADER_SIZE+msg_length:] #Update the input buffer by removing the extracted message
                self.post_message(message) #inserts the message into the input queue
            elif msg_code==MESSAGE_PING:
                self.inbound_buffer=self.inbound_buffer[MESSAGE_HEADER_SIZE:]
            elif msg_code==MESSAGE_REFUSED:
                self.refused=True
                self.close() #closes the channel
                break
            else:
                self.found_error=True
                break




class TransientChannel(BaseChannel):
    def __init__(self, sock, dispatcher):
        self.dispatcher=dispatcher
        self.received_first=False #Flag to indicate whether the first message has been received.
        super().__init__() #Initializes the BaseChannel base class.
        self.set_socket(sock)

    def close(self):
        super().close()
        self.dispatcher.close_channel(self) #Notifies the dispatcher that the channel has been closed, passing the channel reference.

    def on_error(self):
        self.close()

    def post_message(self, message):
        with self.lock:
            if self.received_first:
                super().post_message(message) #Call the base class's post_message method to insert the message into the input queue.
                return
            else:
                self.received_first=True
        self.dispatcher.register_channel(self, message) #Notifies the dispatcher that the channel has been registered, passing the channel reference and the first message.


class ClientChannel(BaseChannel):
    def __init__(self, host, port, hello_message):
        self.host=host
        self.port=port
        self.hello_message=hello_message #Set the welcome message that will be sent to the server after connecting.
        super().__init__()
        self.reconnect() #Attempts to connect to the server by calling the reconnect method.

    #In case of failure, the channel attempts to reconnect to the server.
    def on_error(self):
        self.reconnect()

    def reconnect(self):
        if self.is_refused(): #If the connection was rejected, it exits the method without attempting to reconnect.
            return
        with self.lock:
            if self.sock: #If a socket already exists, try to close it.
                try:
                    self.sock.close()
                finally:
                    self.sock=None
            n=0
            while not self.sock: #It keeps trying to create a new socket until it succeeds.
                try:
                    self.sock=create_client_socket(self.host, self.port) #Attempts to create a new client socket with the specified host and port.
                except IOError: #If an I/O error occurs while creating the socket, increments n (to a maximum of 50) and waits an increasing amount of time before trying again.
                    n=min(n+1, 50)
                    time.sleep(0.1*n)
            self.inbound_buffer=b''
            self.outbound_buffer=self.encode_message(self.hello_message) #It encodes the welcome message and places it in the output buffer to be sent to the server.


class ServerChannel(AbstractChannel):
    def __init__(self, key, dispatcher):
        self.key=key #Unique key to identify the channel.
        self.dispatcher=dispatcher
        self.delegate=None #Delegated channel to perform communication.
        self.lock=threading.RLock()
        self.closed=False
        self.last_time=time.time()

    def get_key(self):
        return self.key

    def set_delegate(self, channel):
        with self.lock:
            if self.delegate: #Closes the previous delegate if it exists
                self.delegate.close()
            self.delegate=channel #Set the new delegate channel.
            self.last_time=time.time()
            if self.closed and self.delegate: #If the channel is marked as closed, it also closes the new delegate and sets it to None.
                self.delegate.close()
                self.delegate=None
                

    #Se esiste un delegato, invia il messaggio tramite il delegato.
    def send(self, message):
        with self.lock:
            if self.delegate:
                self.delegate.send(message)

    #If a delegate exists, send a rejection message via the delegate.
    def send_refuse(self):
        with self.lock:
            if self.delegate:
                self.delegate.send_refuse()

    #If a delegate exists, attempts to receive a message through the delegate.
    def receive(self, timeout=None):
        with self.lock:
            if self.delegate:
                return self.delegate.receive(timeout)
            else:
                return None

    #Closes the delegate if it exists and sets it to None.
    def close(self):
        with self.lock:
            if self.delegate:
                self.delegate.close()
                self.delegate=None
            self.closed=True
        self.dispatcher.close_channel(self)

    #Returns the value of the closed flag.
    def is_closed(self):
        with self.lock:
            return self.closed

    #If a delegate exists, check if it is rejected
    def is_refused(self):
        with self.lock:
            if self.delegate:
                return self.delegate.is_refused()
        return False

    #Returns the closed status of the channel
    def is_finished(self):
        return self.is_closed()

    #If a delegate exists, returns the time of the delegate's last activity.
    #If there is no delegate, returns the time of the last recorded activity.
    def last_activity_time(self):
        with self.lock:
            if self.delegate:
                return self.delegate.last_activity_time()
            else:
                return self.last_time


class Dispatcher:
    def __init__(self, port):
        self.sock=create_server_socket(port) #Creates a server socket that listens on the specified port.
        self.must_finish=False #Indicates whether the dispatcher should terminate.
        self.finished=False #Indicates whether the dispatcher has finished.
        self.error=False #Indicates if an error occurred.
        self.lock=threading.RLock()
        self.transient_channels=set()
        self.server_channels=dict() #Dictionary of active server channels, indexed by key.
        t=threading.Thread(target=self.thread_function)
        t.start()



    #Returns the welcome message as the connection key. This method can be overridden to define different key logic.
    def connection_key(self, hello_message):
        return hello_message

    def register_channel(self, transient_channel, hello_message):
        key=self.connection_key(hello_message)
        if key is None:
            transient_channel.send_refuse()
            self.close_channel(transient_channel)
        else:
            with self.lock:
                if key in self.server_channels:
                    self.server_channels[key].set_delegate(transient_channel)
                else:
                    sc=ServerChannel(key, self)
                    sc.set_delegate(transient_channel)
                    self.server_channels[key]=sc
                    self.on_new_channel(key, sc)


    def on_new_channel(self, key, channel):
        pass

    def on_error(self):
        pass

    def close_channel(self, channel):
        if isinstance(channel, TransientChannel):
            with self.lock:
                if channel in self.transient_channels:
                    self.transient_channels.remove(channel)
        else:
            key=channel.get_key()
            with self.lock:
                if key in self.server_channels:
                    del self.server_channels[key]
        if not channel.is_closed():
            channel.close()

    def get_keys(self):
        with self.lock:
            return set(self.server_channels.keys())

    def get_channel(self, key):
        with self.lock:
            return self.server_channels.get(key, None) 


    def shutdown(self, wait=False):
        with self.lock:
            self.must_finish=True
        if wait:
            while not self.is_finished():
                time.sleep(WAIT_TIME)

    def is_finished(self):
        with self.lock:
            return self.finished

    def is_in_error(self):
        with self.lock:
            return self.error

    def thread_function(self):
        with self.lock:
            must_finish=self.must_finish
        while not must_finish:
            with self.lock:
                fd=self.sock.fileno()
            rready,_,xready=select.select([fd],[],[fd], WAIT_TIME)
            if xready:
                with self.lock:
                    self.must_finish=True
                    self.error=True
                    self.on_error()
            if rready:
                with self.lock:
                    try:
                        if not self.must_finish:
                            sock,addr=self.sock.accept()
                            tc=TransientChannel(sock, self)
                            self.transient_channels.add(tc)
                    except IOError:
                        self.must_finish=True
                        self.error=True
                        self.on_error()
            with self.lock:
                must_finish=self.must_finish
            # End While
        try:
            self.sock.close()
        except IOError as e:
            pass
        self.sock=None
        with self.lock:
            for ch in list(self.transient_channels):
                ch.close()
            for ch in list(self.server_channels.values()):
                ch.close()
            self.transient_channels.clear()
            self.server_channels.clear()
            self.finished=True


def create_server_socket(port):
    return socket.create_server(('',port))


def create_client_socket(host, port):
    return socket.create_connection((host,port))

def encode_float_list(lst):
    try:
        return b''.join((struct.pack('!f', x) for x in lst))
    except:
        return None

def decode_float_list(msg):
    try:
        return list(x[0] for x in struct.iter_unpack('!f', msg))
    except:
        return None
