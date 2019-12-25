# stock1_day1,stock1_day2,...,stock1_day7,stock2_day1,stock2_day2,...,stock8_day7
import pickle
stock1 = [110.8,110.9,118.7,119.1,118.8,118.5,118.7]
stock2 =[49.9, 49.9,55.5,55.6,54.4,54.7,55.1]
stock3 = [49.9,49.9,55.5,55.6,54.4,54.7,55.1]
stock4 = [61.1,60.2,62.3,61.8,62.1,61.8,60.6]
stock5 = [111.2,109.6,106.9,106.4,107.2,106.1,102.9]
stock6 = [876.1,872.8,705.3,709.1,702,715.5,704.3]
stock7 =  [55.9,57.5,43.1,43,43.1,43,43.1]
stock8 = [29.8,29.8,30.1,30.5,30.4,30.2,30.4]
stock = stock1 + stock2 + stock3 +stock4 +stock5 + stock6 + stock7 + stock8
# with open("result.pkl", 'wb') as fo:     # 将数据写入pkl文件
#     pickle.dump(stock, fo)
# with open("result.pkl", 'rb') as fo:
#     list_data = pickle.load(fo, encoding='bytes')
#
# print(list_data)
