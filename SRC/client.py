import channel
import numpy as np

JOINTS=11
STATE_DIMENSION=37
DEFAULT_PORT=9543

class Client:
    def __init__(self, name, host='localhost', port=DEFAULT_PORT):
        key=name.encode('utf8') #UTF-8 encoded version of the client name.
        self.channel= channel.ClientChannel(host, port, key)

    def get_state(self, blocking=True):
        timeout=None
        if blocking: #the method will wait until it receives a message (infinite timeout).
            timeout=-1 #infinite timer
        last_msg=self.channel.receive(timeout)
        if last_msg is None:
            return None
        msg=self.channel.receive()
        while msg is not None:
            last_msg=msg
            msg=self.channel.receive()
        state= channel.decode_float_list(last_msg)
        return np.array(state)

    def send_joints(self, joints):
        joints=list(joints)
        if len(joints)!=JOINTS:
            raise ValueError('Unvalid number of elements')
        msg= channel.encode_float_list(joints) #Decode the last message into a list of floats.
        if msg is None:
            raise ValueError('Unvalid joints vector')
        self.channel.send(msg)

    def close(self):
        self.channel.close()
