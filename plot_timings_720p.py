import matplotlib.pyplot as plt

barlist=plt.bar(['Transmission \n start to processing', 'Round-trip time', 'Inference', 'Result drawing', 'Transmission \n to client'], 
                [0.0020, 0.014, 0.013, 0.0049, 0.0015])
barlist[0].set_color('royalblue')
barlist[1].set_color('tomato')
barlist[2].set_color('seagreen')
barlist[3].set_color('orange')
barlist[4].set_color('mediumpurple')
plt.xlabel('', fontsize=20)
plt.ylabel('time (s)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

#ax.set_xticklabels([])
