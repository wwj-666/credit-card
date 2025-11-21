import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import re
from imutils import contours
from imutils import resize
import myutils  

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#读取模板图像
img =cv2.imread('model.png',cv2.IMREAD_COLOR)
cv_show("template",img)   
#对模板图像取灰度图
ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show("ref",ref)
#二值化
ref=cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
cv_show("thresh",ref)

#计算轮廓
#cv2.RETR_EXTERNAL表示只检测外轮廓
#cv2.CHAIN_APPROX_SIMPLE表示只保存轮廓的端点信息
#cv2.findContours()函数接收的参数为二值图
#ref_是 - 处理后的二值图像refCnts - 检测到的轮廓点集列表hierarchy - 轮廓间的层级关系信息
refCnts,hierarchy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,refCnts,-1,(0,0,255),3)
print("轮廓总数:", len(refCnts))
cv_show("contours",img)


#对轮廓进行排序，保证从左到右的顺序#[0]: 排序后的轮廓列表(refCnts)
refCnts =contours.sort_contours(refCnts,method="left-to-right")[0]   
digits = {}
#遍历每一个轮廓
for (i,c) in enumerate(refCnts):
    #计算外接矩形#求出轮廓的边界框坐标
    (x,y,w,h) = cv2.boundingRect(c)
    #提取出每一个数字的ROI就是想要截取的区域
    roi =ref[y:y+h,x:x+w]
    roi =cv2.resize(roi,(57,88))
    #对每个roi进行保存
    digits[i] =roi
    #展示每个roi
    cv_show("roi_{}".format(i),roi)


#初始化卷积核
rectKernel =cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#读取待识别的信用卡图像 预处理
image =cv2.imread("kard.png")
cv_show("image",image)
height, width = image.shape[:2]
print(f"原始图像尺寸: 宽={width}, 高={height}")  # 调试信息
aspect_ratio = width / height
new_width = 300
new_height = int(new_width / aspect_ratio)
image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show("gray", gray)
#礼帽变换突出亮细节文字识别中增强字符特征

tophat =cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv_show("tophat",tophat)
#dx=1, dy=0: 计算x方向的一阶导数（水平边缘检测 ksize=-1: 使用Scharr算子（比3x3 Sobel算子更精确）
gradX =cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
#整体功能是将梯度图像标准化为0-255范围内的uint8类型。
gradX =np.absolute(gradX)
(minVal,maxVal) = (np.min(gradX),np.max(gradX))
# 添加的归一化步骤
gradX =((gradX - minVal) / (maxVal - minVal)) * 255
gradX =gradX.astype("uint8")
#这里是在打印 gradx 变换后得到的 NumPy 数组的形状信息。
print(np.array(gradX).shape)
#通过闭操作 连接字符区域先膨胀后腐蚀）
gradX =cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
cv_show("gradX",gradX)

thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show("thresh",thresh)
#再次通过闭操作先膨胀后腐蚀因为白色的块中有空
thrsh =cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)

#计算轮廓
#cv2.RETR_EXTERNAL只检索最外层轮廓 cv2.CHAIN_APPROX_SIMPLE压缩水平、垂直和对角线段，只保留端点 返回轮廓数组refCnts和层级结构hierarchy
threshCnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
#在原图上画出所有的轮廓
cur_img =image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show("contours",cur_img)
#筛选符合信用卡号码区域的轮廓
locs = []
#创建了一个元组(tuple)，包含两个元素：变量i和变量c enumerate 是Python内置函数，用于将一个可迭代对象（如列表、元组等）组合为索引序列
for (i,c) in enumerate(cnts):
    #计算外接矩形,利用长宽比筛选出符合信用卡号码区域的轮廓 cv2.boundingRect(c) 计算其最小外接
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    if ar > 2.5 and ar < 4.0: # 调整宽高比范围 
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x,y,w,h))  
            
#将符合的轮廓从小到大排序排序是根据 x[0] 进行的，这里的 x 代表 locs 中的每个元素。从上下文推断，locs 中存储的应该是轮廓的边界框坐标 (x, y, w, h)，其中 x[0] 就是每个边界框的 x 坐标（左上角横坐标）
locs =sorted(locs,key=lambda x:x[0])    
output =[]

#遍历每一个轮廓中的数字区域
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    groupOutput =[]

    #根据坐标提取每一个组
    group =gray[gY-5:gY+gH+5,gX-5:gX+gW+5]
    cv_show("group",group)
    #得到每一组后，再组内进行轮廓检测
    #预处理
    group =cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show("group",group)
    group_digitCnts,hierarchy =cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours (group_digitCnts,method="left-to-right")[0]

    #遍历每一个数字轮廓
    for c in digitCnts:
        #计算外接矩形
        (x,y,w,h) =cv2.boundingRect(c)
        roi =group[y:y+h,x:x+w]
        roi =cv2.resize(roi,(57,88))
        cv_show("roi",roi)

        #计算机匹配得分
        scores =[]

        #在模板中计算每一个得分 返回字典中所有键值对的视图对象，每个键值对以元组形式(key, value)呈现。
        for (digit, digitROI) in digits.items():
        #模板匹配
            result =cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
        #只返回最大值
            (_,score,_,_) =cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax (scores)))  
#画出来
    cv2.rectangle (image,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),1)    
    cv2.putText(image,''.join(groupOutput),(gX,gY-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
    output.extend (groupOutput)

#打印结果
   
# 将output中的每个numpy.int64转换为str
print("信用卡号码: {}".format(''.join(str(digit) for digit in output)))                    
cv_show("image",image)
