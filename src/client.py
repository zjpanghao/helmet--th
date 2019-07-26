#!/usr/bin/env python
import sys, glob

import Helmet
from ttypes import HelmetResult
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

try:
    transport = TSocket.TSocket('localhost', 9090)
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = Helmet.Client(protocol)

    transport.open()

    result = client.checkHelmet('Kongxx')
    print(result)

    transport.close()
except Thrift.TException as tx:
    print (tx.message)
