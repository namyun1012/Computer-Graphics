#!/usr/bin/env python3
# -*- coding: utf-8 -*
# sample_python aims to allow seamless integration with lua.
# see examples below

import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image 

# Color Support, Not need to change
# 약간의 조정 정도만 있음 될 듯?
class Color:
    def __init__(self, R, G, B):
        self.color=np.array([R,G,B]).astype(np.float64)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)

class Surface:
    def __init__(self, type, ref, center, radius):
        self.type = type
        self.ref = ref
        self.center = center
        self.radius = radius
        

class Shader:
    def __init__(self, name, type, diffuse, specular, exponent):
        self.name = name
        self.type = type
        self.diffuse = diffuse
        self.specular = specular
        self.exponent = exponent
        
class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity




# r(t) = p + td
# view point = p 
# hitSurface 랑 t를 반환함
# d가 unit vector 이므로 t는 거리이다.
# hitSurface는 index로 처리함
# 지금 현재는 surface가 영향을 주지 못하고 있다.
def sphere_intersect(d, view_point, surface_list):
    hitSurface = -1
    tBest = np.inf
    cnt = 0
    
    # 각 surface 별로 어디 surface에 부딪히는 지 확인
    for surface in surface_list:
        center = surface.center
        radius = surface.radius
        
        distance = view_point - center # view_point 에서 surface의 center 까지의 거리
        a = np.dot(d, d) # d.d
        b = 2.0 * np.dot(distance, d) # distance. d
        c = np.dot(distance, distance) - radius * radius # distance . distance - d.d 
        
        
        temp = b * b - 4 * a * c # 확인용 연산
        
        
        # tBest 결정하는 작업, 0 보다는 커야 함
        if temp >= 0:
            tmin = (-b - np.sqrt(temp)) / (2.0 * a)
            tplus = (-b + np.sqrt(temp)) / (2.0 * a)

            if tmin > 0 and tmin < tBest:
                tBest = tmin
                hitSurface = cnt
            
            if tplus > 0 and tplus < tBest:
                tBest = tplus
                hitSurface = cnt    
               
        cnt += 1

    
    return hitSurface, tBest

def shading(hitSurface, t, surface_list, light_list, shader_list, view_point, d , w_vec):
      
    result = np.zeros(3)
    
    if t == np.inf:
        return result
    
    #for surface in surface_list:
        
    shader = shader_list[hitSurface]
        
    
    color = shader.diffuse
    specular = shader.specular
    exponent = shader.exponent
        
    type = shader.type

    # 사실상 1번만 돔
    for light in light_list:
        light_position = light.position
        light_intensity = light.intensity
        intersection_point = view_point + t * d
        
            
            # right_dir = 
        light_dir = light_position - intersection_point
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        
        
        # 빛이 사물에 막히는 지 확인 필요
        check_surface, temp = sphere_intersect(-light_dir, light_position, surface_list)
        if check_surface != hitSurface :
            return result
        
        center = surface_list[check_surface].center
        radius = surface_list[check_surface].radius
        
        n = (intersection_point - center) / radius
        n /= np.linalg.norm(n)
        
        
        # normal_vector w_vec과 light_vector light_dir 를 dot product 연산함
        # Ld = Kd I max(0, n.I)
        # n을 새로 구하자
        theta = np.dot(light_dir, n)
            
        diffuse = np.zeros(3)
        
        # light_intensity * theta * shader_color
        if theta > 0:
            diffuse = light_intensity * theta * color
                
            
            
                
        result += diffuse
        
        # type이 Phong일 때만 수행하는 연산
        # specular color 또한 계산해야 한다. 반사광 고려 필요
        if type == 'Phong':
            
            #h = 2 * theta * (w_vec) - light_dir
            
            # n = w_vec   
            #alpha = np.dot(h, -d)
            # h = l + v
            
            h = light_dir - d
            h /= np.linalg.norm(h)
            
            alpha = np.dot(n, h)
            
            
            specular_result = np.zeros(3)    
            
            specular_result =  specular * light_intensity * (max(alpha, 0) ** exponent)
                
            result += specular_result
        
    
    return result

def main():


    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    # set default values
    viewDir=np.array([0,0,-1]).astype(np.float64)
    viewUp=np.array([0,1,0]).astype(np.float64)
    viewProjNormal=-1*viewDir  # you can safely assume this. (no examples will use shifted perspective camera)
    viewWidth=1.0
    viewHeight=1.0
    projDistance=1.0
    intensity=np.array([1,1,1]).astype(np.float64)  # how bright the light is.
    position = np.array([1,1,1]).astype(np.float64) # 임시값
    print(np.cross(viewDir, viewUp))

    imgSize=np.array(root.findtext('image').split()).astype(np.int32)
    
    # lists
    shader_list =[]
    surface_list = []
    light_list = [] # light 도 여러개가 존재하는 경우가 있다.
    
    # parsing camera viewPoint = Camera
    # camera는 1개임
    for c in root.findall('camera'):
        viewPoint=np.array(c.findtext('viewPoint').split()).astype(np.float64)
        
        # 내가 추가한 parsing code
        viewDir = np.array(c.findtext('viewDir').split()).astype(np.float64)
        viewUp = np.array(c.findtext('viewUp').split()).astype(np.float64)
        if(c.findtext('projNormal')):
            viewProjNormal = np.array(c.findtext('projNormal').split()).astype(np.float64)
        if(c.findtext('projDistance')):
            projDistance = np.array(c.findtext('projDistance')).astype(np.int64)
            
        viewWidth = np.array(c.findtext('viewWidth').split()).astype(np.float64)
        viewHeight = np.array(c.findtext('viewHeight').split()).astype(np.float64)
        
        
        print('viewpoint', viewPoint)
        print(viewDir)
        print(viewProjNormal)
        print(viewWidth)
        print(viewHeight)
        print(projDistance)
        
    #diffuseColor_c = 
    #shader는 여러 개인 경우 존재함
    for c in root.findall('shader'):
        specularColor = None
        exponent = None
        
        diffuseColor_c=np.array(c.findtext('diffuseColor').split()).astype(np.float64)
        if(c.findtext('specularColor')):
            specularColor = np.array(c.findtext('specularColor').split()).astype(np.float64)
        if(c.findtext('exponent')):
            exponent = np.array(c.findtext('exponent')).astype(np.int64)
        name = c.get('name')
        type = c.get('type')
        
        print('name', name)
        print('type', type)
        print('diffuseColor', diffuseColor_c)
        print('exponent', exponent)
        print('specularColor', specularColor)
        
        new_shader = Shader(name, type, diffuseColor_c, specularColor, exponent)
        shader_list.append(new_shader)
        #추가 Parsing Code
    
    #surface 찾아서 넣기
    for c in root.findall('surface'):
        type = c.get('type')
        
        #shader ref  찾기
        for d in c.findall('shader'):
            ref = d.get('ref')
        
        center = np.array(c.findtext('center').split()).astype(np.float64)
        radius = np.array(c.findtext('radius')).astype(np.int64)
            
        print('type', type)
        print('ref', ref)
        print('center', center)
        print('radius', radius)
        
        surface_list.append(Surface(type, ref, center, radius))

    #마지막으로 light 찾기
    for c in root.findall('light'):
        position = np.array(c.findtext('position').split()).astype(np.float64)
        intensity = np.array(c.findtext('intensity').split()).astype(np.float64)
        print('position', position)
        print('intensity', intensity)
        
        light_list.append(Light(position, intensity))
        
        
    print(len(surface_list))
    print(len(shader_list))   
    print(len(light_list))   
    #code.interact(local=dict(globals(), **locals()))  

    # Create an empty image -> not need to change
    channels=3
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
    img[:,:]=0
    
    white=Color(1,1,1).toUINT8()
    red=Color(1,0,0).toUINT8()
    blue=Color(0,0,1).toUINT8()
    
    # Basic Vectors
    w_vec = -viewDir # 이미지 쪽으로 가는 벡터
    w_vec = w_vec / np.linalg.norm(w_vec) # unit vector 로
    
    u_vec = np.cross(-w_vec, viewUp) # 이미지 평면의 벡터 u_vec, v_vec
    v_vec = np.cross(-w_vec, u_vec)
    
    # unit vector 화
    
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = v_vec / np.linalg.norm(v_vec)
    
    
    #My Code, 계산한 이미지(색)을 좌표에 넣어줌
    for y in np.arange(imgSize[1]):
       for x in np.arange(imgSize[0]):
        #first, compute viewing ray
        # r(t) = p + td (p : viewpoint)
        
        # 우선 pixel to image 방법 사용, u, v의 좌표를 구함 u_vec 과 v_vec이 반대인듯?
        # width의 절반을 빼줌 
        v = viewWidth * (x + 0.5) / imgSize[0] - (viewWidth / 2)
        u = viewHeight * (y + 0.5) / imgSize[1] - (viewHeight / 2)
        
        #이미지 위의 점의 좌표 s
        s = u * u_vec + v * v_vec + viewPoint - w_vec * projDistance # -dW
        
        # d를 구함 이후 unit vector로 만듬
        d = s - viewPoint
        d = d / np.linalg.norm(d)
        
        
        # p(viewPoint) 와 d를 사용해서 t를 구함
        # find first object hit by ray and its surface normal n
        hitsurface, t = sphere_intersect(d, viewPoint, surface_list)
        #if hitsurface > 0:
        #    print(hitsurface)
       
        result =  shading(hitsurface, t, surface_list, light_list, shader_list, viewPoint, d, w_vec)
        
        img[x][y] = Color(result[0], result[1], result[2]).toUINT8()
        
        
    
    rawimg = Image.fromarray(img, 'RGB')
    #rawimg.save('out.png')
    rawimg.save(sys.argv[1]+'.png')
    
if __name__=="__main__":
    main()
