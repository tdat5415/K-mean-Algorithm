# K-mean-Algorithm

python의 numpy 라이브러리를 이용하여 K mean Algorithm을 쉽게 구현하는 방법을 다룬다.

---

#### 그룹 데이터 생성

```python
import matplotlib.pyplot as plt
import numpy as np

g_1_centroid = np.array([20, 30])
group_1 = np.random.randn(1000, 2) * 2 + g_1_centroid[None, :]

g_2_centroid = np.array([25, 35])
group_2 = np.random.randn(500, 2) * 1 + g_2_centroid[None, :]

g_3_centroid = np.array([20, 40])
group_3 = np.random.randn(500, 2) * 1 + g_3_centroid[None, :]

# plt.plot(group_1[:,0], group_1[:,1], marker='o', color='#%02x%02x%02x' % (255, 0, 0))
# plt.plot(group_2[:,0], group_2[:,1], marker='o', color='#%02x%02x%02x' % (0, 255, 0))
# plt.plot(group_3[:,0], group_3[:,1], marker='o', color='#%02x%02x%02x' % (0, 0, 255))
plt.plot(group_1[:,0], group_1[:,1], 'ro')
plt.plot(group_2[:,0], group_2[:,1], 'go')
plt.plot(group_3[:,0], group_3[:,1], 'bo')
plt.show()
```

![038c23f2-3f63-4096-aebd-6dd9b9ca5393](https://user-images.githubusercontent.com/48349693/156320429-c46c5fac-ef4d-454a-82e4-f3a6ec7e7eeb.png)

먼저 데이터를 생성한다. 위 그림에서는 세 개의 그룹으로 되어있다.  
위의 주석 부분은 출력할 때 점 사이에 선이 생겨 보기가 좋지 않다.

---

#### K mean Algorithm

```python
import numpy as np

def get_centroids(all_dot, num=3, epoch=10):
    all_dot = np.array(all_dot) # (2000, 2)
    np.random.shuffle(all_dot)
    next_centroids = all_dot[:num] # (num,2)
    for _ in range(epoch):
        dist_all_to_cent = np.linalg.norm(all_dot[:,None,:] - next_centroids[None,...], axis=-1) # (2000, num)
        min_idxs = np.argmin(dist_all_to_cent, axis=-1) # (2000,)
        unique, counts = np.unique(min_idxs, return_counts=True) # (3,), (3,) # unique is auto sorted
        seleted = np.stack([np.where(min_idxs[:,None]==i, all_dot, 0) for i in range(num)]) # (3, 2000,2) # many zeros
        tot_each = np.sum(seleted, axis=1) # (3,2)
        next_centroids = tot_each / counts[:,None] # (3,2)
    return next_centroids
```

알고리즘 부분의 함수이다.  
unique 리스트는 여기서 쓰이지 않았지만 출력을 찍어보면 numpy에서 알아서 sort되서 나온다.  
그래서 range(num)을 사용하였다. 그냥 그 부분에 unique 리스트 사용할 걸 그랬다.  
counts는 각 총합에서 나눠줌으로써 평균을 구하기 위해서 필요하다.  
참고로 다차원 점도 적용 가능하다.

---

#### 점들 분류하기

```python
import numpy as np

def split_all_dot(all_dot, centroids):
    dist_all_to_cent = np.linalg.norm(all_dot[:,None,:] - centroids[None,...], axis=-1) # (2000, num)
    min_idxs = np.argmin(dist_all_to_cent, axis=-1) # (2000,)
    return [all_dot[np.where(min_idxs==i)] for i in range(len(centroids))]
```

중앙이 되는 좌표들은 이미 구한 상태이다.  
여기서 분류가 되지 않은 점들을 centroids로 분류한다.

---

#### 실행

```python
import numpy as np

all_dot = np.concatenate([group_1, group_2, group_3], axis=0)
centroids = get_centroids(all_dot, num=3, epoch=10)
split_dots = split_all_dot(all_dot, centroids)
```

---

#### 확인하기

```python
import matplotlib.pyplot as plt

c = ['r', 'g', 'b']
for i, dots in enumerate(split_dots):
    plt.plot(dots[:,0], dots[:,1], f'{c[i]}o', )
plt.show()
```

![07a6b654-b54c-411f-8f6e-1862ab4accbb](https://user-images.githubusercontent.com/48349693/156322497-58813bea-eded-4b3c-ab9d-de61859d7819.png)

맨 위의 그림과는 색깔이 다를뿐 중앙 좌표를 기준으로 색깔별로 분류하여 나타내었다.  
중앙 좌표들을 적절히 구한 것 같다.




