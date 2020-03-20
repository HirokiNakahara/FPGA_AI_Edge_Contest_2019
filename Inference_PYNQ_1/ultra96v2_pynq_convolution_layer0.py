
# coding: utf-8

# ## Convolutional Operation Demo<br>PYNQ on Ultra96v2 board  
# 
# H. Nakahara (Tokyo Tech.) 19th/Mar./2020  
# Copyright all rights reserved.

# ### Setup
# load bitstream file

# In[40]:


from pynq import Overlay
import pynq

overlay = Overlay('/home/xilinx/pynq/overlays/base/conv_l0_2.bit')
#dir(overlay)


# In[41]:


registers = overlay.kernel_0.register_map


# load testbench file

# In[42]:


import numpy as np

inimg = np.loadtxt('/home/xilinx/data/testbench_input.txt')


# In[43]:


inimg = inimg.reshape((3,416,416)).transpose(1,2,0) # Y,X,CH
inimg = inimg * 1024.0
inimg = inimg.astype(np.int32)


# Setup DMA buffer

# In[44]:


import pynq.lib.dma

dma = overlay.axi_dma_0


# In[45]:


from pynq import Xlnk

inimg_size = 416*11*3
outfmap_size = 102*64+1

xlnk = Xlnk()

send_buf   = xlnk.cma_array(shape=(inimg_size),dtype=np.int32)
recv_buf = xlnk.cma_array(shape=(outfmap_size),dtype=np.int32)


# In[46]:


inimg_buf   = np.zeros((11,416,3)).astype(np.int32)
outfmap_buf = np.zeros((102,64,102)).astype(np.int32)


# ### Perform Convolutional Operation (...but too slow)

# In[ ]:


get_ipython().run_cell_magic('time', '# performance counter', 'for line in range(102):\n    # load input image\n    for i in range(11):\n        inimg_buf[i] = inimg[i+line*4]\n    \n    tmp = inimg_buf.copy().transpose((2,0,1)).reshape(-1,) # CH,Y,X\n    send_buf[0:inimg_size] = tmp[0:inimg_size]\n\n    # activate DMA\n    registers.CTRL.AP_START = 1\n\n    # DMA access\n    dma.sendchannel.transfer(send_buf)\n    dma.recvchannel.transfer(recv_buf)\n\n    # wait DMA\n    dma.sendchannel.wait()\n    dma.recvchannel.wait()\n    \n    # store output buffer\n    tmp2 = recv_buf[0:outfmap_size - 1]\n    tmp2 = tmp2.reshape((64,102)) # CH, X\n    outfmap_buf[line] = tmp2')


# ### Verification with C++ testbench

# In[34]:


outfmap_buf = outfmap_buf.transpose((1,0,2)) / 1024.0 # Y,CH,X -> CH,Y,X


# In[36]:


bench_outfmap = np.loadtxt('/home/xilinx/data/testbench_output.txt')


# In[39]:


error = np.abs(bench_outfmap - outfmap_buf.reshape(-1,))
max_error = np.max(error)

print('max error',max_error)

if max_error < 0.1:
    print('TEST_PASS')
else:
    print('TEST_FAILURE')

