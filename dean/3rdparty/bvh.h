#ifndef BVH_H
#define BVH_H

#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

/*

todo,
这个 bvh tree应该是okay了, 过一边, 理解. 看是否又bug

不要在31.1上改, 创建31.2, 面试完了有空在弄.
总结一下, 现在我build BVH tree成功了, 后面要传3个SSBO 到CS, BVHNodes array, Primitives array和Spheres array, 然后在CS里面根据这3个ssbo 每个生成的ray都遍历这个tree., hit 到node了, 把里面的primitives拿出来和ray hit一下.

*/

struct Sphere {
    glm::vec3 center;
    float radius;
    // int materialId;  // 如果你已经有材质的话
};

// 如果是叶子 (primitiveCount > 0),
//     当前AABB包含了primitiveCount个primitive, 从 primitives[leftChildOrFirstPrimitive]开始
// 如果是内部节点 (primitiveCount == 0),
// //     左子节点为bvhNodes[leftChildOrFirstPrimitive], 右子节点为bvhNodes[leftChildOrFirstPrimitive + 1]
// struct BVHNode {
//     glm::vec3 aabbMin;
//     // bvhNodes里的idx或者被partition后的primitives的idx
//     int leftChildOrFirstPrimitive;
//     glm::vec3 aabbMax;
//     // =0 表示内部节点, >0 表示 leaf 节点, 即当前AABB中 primitive 的个数.
//     int primitiveCount;
// };

struct BVHNode {
    glm::vec3 aabbMin;
    int leftChild;  // 对于内部节点，表示左子节点索引；对于叶子节点，未使用

    glm::vec3 aabbMax;
    int rightChild;  // 对于内部节点，表示右子节点索引；对于叶子节点，未使用

    int primitiveOffset;  // 对于叶子节点，表示 primitives[] 的起始位置；对于内部节点，未使用
    int primitiveCount;   // == 0 表示内部节点；> 0 表示叶子节点，表示 primitive 个数
};

inline bool isLeaf(const BVHNode& node) {
    return node.primitiveCount > 0;
}

// cs里面也要一个一样的.
struct PackedBVHNode {
    glm::vec4 aabbMin;  // xyz: aabb.min, w: leftChild
    glm::vec4 aabbMax;  // xyz: aabb.max, w: rightChild
    glm::ivec4 meta;    // x: primitiveOffset, y: primitiveCount, z: unused, w: unused
};


struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB() {
        min = glm::vec3(std::numeric_limits<float>::max());
        max = glm::vec3(std::numeric_limits<float>::lowest());
    }

    AABB(const glm::vec3& mi, const glm::vec3& ma)
        : min(mi), max(ma) {}

    void expand(const AABB& other) {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    void expand(const glm::vec3& p) {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }

    // 返回质心, 即这个AABB的中心坐标
    glm::vec3 centroid() const {
        return 0.5f * (min + max);
    }
};

AABB computeSphereAABB(const Sphere& s) {
    return {
        s.center - glm::vec3(s.radius),
        s.center + glm::vec3(s.radius)};
}

class BVH {
public:
    BVH() { makeExampleSpheres(); }
    ~BVH() {}
    void makeExampleSpheres() {

        // 测试, 在运行到的任意地方用: BVH bvh;

        // 在z上排列, 无序
        float r = 0.2f;
        std::vector<Sphere> spheres = {
            {glm::vec3(0.0f, 0.0f, 2.5f), r},
            {glm::vec3(0.0f, 0.0f, 1.5f), r},
            {glm::vec3(0.0f, 0.0f, 6.0f), r},
            {glm::vec3(0.0f, 0.0f, 1.0f), r},
            {glm::vec3(0.0f, 0.0f, 3.0f), r},
            {glm::vec3(0.0f, 0.0f, 2.0f), r},
        };

        primitives.resize(spheres.size());
        std::iota(primitives.begin(), primitives.end(), 0);
        bvhNodes.clear();
        int rootIndex = buildBVHRecursive(0, primitives.size(), spheres);
        std::cout << "[BUILD DONE] root index = " << rootIndex << std::endl;

        printBVH();
        /*
3 1 5 0 4 2
node 0 [INNER] left=1 right=6
node 1 [INNER] left=2 right=3
node 2 [LEAF]  offset=0 count=2
node 3 [INNER] left=4 right=5
node 4 [LEAF]  offset=2 count=1
node 5 [LEAF]  offset=3 count=2
node 6 [LEAF]  offset=5 count=1

             node 0
            /      \
        node 1     node 6
       /     \
   node 2   node 3
             / \
         node 4 node 5



        */

        // BVH bvh;
        // std::vector<PackedBVHNode> gpuNodes = bvh.flattenBVH();
        // std::vector<int> gpuPrims = bvh.getPrimitives();
        std::vector<PackedBVHNode> gpuNodes = flattenBVH();
        std::vector<int> gpuPrims = getPrimitives();

        std::cout << "Node[0] AABB: " << glm::to_string(gpuNodes[0].aabbMin) << ", "
                  << glm::to_string(gpuNodes[0].aabbMax) << std::endl;
    }

private:
    std::vector<BVHNode> bvhNodes;
    // primitives 是一个数组，里面是 球array spheres的索引
    std::vector<int> primitives;

    std::vector<PackedBVHNode> flattenBVH() const {
        std::vector<PackedBVHNode> out;
        for (const auto& node : bvhNodes) {
            PackedBVHNode p;
            p.aabbMin = glm::vec4(node.aabbMin, static_cast<float>(node.leftChild));
            p.aabbMax = glm::vec4(node.aabbMax, static_cast<float>(node.rightChild));
            p.meta = glm::ivec4(node.primitiveOffset, node.primitiveCount, 0, 0);
            out.push_back(p);
        }
        return out;
    }

    const std::vector<int>& getPrimitives() const {
        return primitives;
    }

    void printBVH() {
        for (size_t i = 0; i < primitives.size(); ++i) {
            std::cout << primitives[i] << " ";
        }
        std::cout << std::endl;

        for (size_t i = 0; i < bvhNodes.size(); ++i) {
            const auto& node = bvhNodes[i];
            if (isLeaf(node)) {
                std::cout << "node " << i << " [LEAF]  offset=" << node.primitiveOffset
                          << " count=" << node.primitiveCount << "\n";
            } else {
                std::cout << "node " << i << " [INNER] left=" << node.leftChild
                          << " right=" << node.rightChild << "\n";
            }
        }
    }

    // 输入: spheres array, primitives array
    // 输出: bvhNodes
    //
    // 这个函数就是对primitives array里面startIndex开始的count个球做:
    // 1. 计算整包围count个球的AABB
    // 2, 得到这个AABB的最长轴axis, x/0, y/1, z/2, 并得到最长轴的中点splitPos
    // 3, 用splitPos对primitives array做一次partition, 球的位置比 splitPos 小的放左边, 大的放右边.
    //     注意这里用户data spheres array不会动, 改变的是primitives array里的值, 也就是指向spheres array里面的球索引.
    // 4, 递归调用 buildBVHRecursive 直到count<=2, 结束
    int buildBVHRecursive(int startIndex, int count, std::vector<Sphere>& spheres) {
        std::cout << ">> dean1 buildBVHRecursive(start=" << startIndex
                  << ", count=" << count << ") node " << bvhNodes.size() << std::endl;

        AABB bounds;
        for (int i = startIndex; i < startIndex + count; ++i) {
            bounds.expand(computeSphereAABB(spheres[primitives[i]]));
        }

        int currentIndex = static_cast<int>(bvhNodes.size());
        bvhNodes.emplace_back();
        // BVHNode& node = bvhNodes.back();
        // BVHNode& node = bvhNodes[currentIndex];

        bvhNodes[currentIndex].aabbMin = bounds.min;
        bvhNodes[currentIndex].aabbMax = bounds.max;

        if (count <= 2) {
            // 创建叶子节点
            bvhNodes[currentIndex].primitiveCount = count;
            bvhNodes[currentIndex].primitiveOffset = startIndex;
            return currentIndex;
        }

        // 内部节点
        bvhNodes[currentIndex].primitiveCount = 0;

        glm::vec3 extent = bounds.max - bounds.min;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        float splitPos = bounds.centroid()[axis];

        auto mid = std::partition(
            primitives.begin() + startIndex,
            primitives.begin() + startIndex + count,
            [&](int idx) {
                AABB aabb = computeSphereAABB(spheres[idx]);
                return aabb.centroid()[axis] < splitPos;
            });

        int midIndex = static_cast<int>(mid - primitives.begin());

        std::cout << "partition axis = " << axis
                  << ", midIndex = " << midIndex
                  << ", start = " << startIndex
                  << ", end = " << (startIndex + count) << std::endl;

        if (midIndex == startIndex || midIndex == startIndex + count) {
            std::cout << "⚠️ partition fallback triggered, force split at "
                      << (startIndex + count / 2) << std::endl;
            // fallback: 强制切
            midIndex = startIndex + count / 2;
            std::rotate(primitives.begin() + startIndex,
                        primitives.begin() + midIndex,
                        primitives.begin() + startIndex + count);
        }

        // 递归构建左右子树
        int leftChild = buildBVHRecursive(startIndex, midIndex - startIndex, spheres);
        int rightChild = buildBVHRecursive(midIndex, startIndex + count - midIndex, spheres);

        bvhNodes[currentIndex].leftChild = leftChild;
        bvhNodes[currentIndex].rightChild = rightChild;

        std::cout << "<< return nodeIndex = " << currentIndex << std::endl;
        assert(midIndex > startIndex && midIndex < startIndex + count && "midIndex fallback failed");
        assert(leftChild != rightChild && "left and right child identical");

        return currentIndex;
    }
};


/*

todo, comment提取回来,

    void buildBVHRecursive(int startIndex, int count, std::vector<Sphere>& spheres) {
        std::cout << "[ENTER] buildBVHRecursive(start=" << startIndex
                  << ", count=" << count
                  << "), bvhNodes.size()=" << bvhNodes.size()
                  << std::endl;

        BVHNode node;
        AABB bounds;

        // 计算三个球的AABB
        for (int i = startIndex; i < startIndex + count; ++i)
            bounds.expand(computeSphereAABB(spheres[primitives[i]]));

        // 第一个node肯定是root.
        node.aabbMin = bounds.min;
        node.aabbMax = bounds.max;

        int currentIndex = bvhNodes.size();
        bvhNodes.push_back(BVHNode{});  // 先占位
        std::cout << "[NEW NODE] index=" << currentIndex << std::endl;

        // count是当前AABB里面球/物体个数, 如果小于等于2, 说明是叶子节点, 否则就是中间节点.
        // 继续划分没意义, BVH 是为了加速遍历时排除无用区域，如果一个节点只有两个 primitive，再分一次，就会变成：
        // 一个子节点包含 1 个 primitive
        // 另一个子节点也包含 1 个 primitive
        // 那么遍历 BVH 时你还得：
        // 先访问内部节点
        // 然后递归访问两个 leaf
        // 这比你直接对两个物体各自做 intersection 要多出 1 次栈访问 + 1 个分支判断，反而变慢了
        // 设置一个终止条件如 count <= 2 是常见手段。你也可以设成 count <= 4，也有的用 maxDepth 来限制
        if (count <= 2) {
            // 叶子节点
            bvhNodes[currentIndex].leftChildOrFirstPrimitive = startIndex;
            bvhNodes[currentIndex].primitiveCount = count;
            return;
        }

        // inner node
        bvhNodes[currentIndex].primitiveCount = 0;

        // 选择分裂轴：最长轴
        glm::vec3 extent = bounds.max - bounds.min;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        // 根据质心分裂, 就是AABB里面最长轴的中点.
        float splitPos = bounds.centroid()[axis];
        // 指定primitives的开始位置和结束位置half-open. 即把primitives过一遍, 按splitPos分成两半. 小的在左边, 大的在右边.
        // 注意这个partition会把primitives里的值改掉, 返回的是迭代器.
        auto mid = std::partition(primitives.begin() + startIndex, primitives.begin() + startIndex + count,
                                  [&](int idx) {  // idx为primitives的值, 不是地址.
                                      // 计算每个球的AABB
                                      AABB aabb = computeSphereAABB(spheres[idx]);
                                      // 如果质心在分裂轴的左边，就返回true
                                      return aabb.centroid()[axis] < splitPos;
                                  });

        int midIndex = mid - primitives.begin();

        // 如果分裂失败，强制一半一半
        if (midIndex == startIndex || midIndex == startIndex + count) {
            std::cout << "[PARTITION FAIL] fallback to mid = " << (startIndex + count / 2)
                      << ", start=" << startIndex << ", count=" << count << std::endl;
            midIndex = startIndex + (count / 2);
        }

        // 先记住左子树构建前的 index
        int leftChildIdx = bvhNodes.size();
        buildBVHRecursive(startIndex, midIndex - startIndex, spheres);

        std::cout << "[RECURSE] left=[" << startIndex << ", " << (midIndex - startIndex) << "], "
                  << "right=[" << midIndex << ", " << (startIndex + count - midIndex) << "]"
                  << std::endl;

        int rightChild = bvhNodes.size();
        buildBVHRecursive(midIndex, startIndex + count - midIndex, spheres);

        assert(rightChild == leftChild + 1); // debug check
        // ⚠️ 现在再更新 leftChild 指针
        bvhNodes[currentIndex].leftChildOrFirstPrimitive = leftChildIdx;
    }




    CS里面大概这样:


好的，下面是一个完整可用的 **GLSL Compute Shader 片段**，专注于 **每条 ray 遍历 BVH 树结构并测试是否击中 sphere**。你可以把它集成到 `.comp` 文件中使用。

---

## ✅ 假设你已有绑定：

```glsl
layout(std430, binding = 0) readonly buffer BVHNodes {
    PackedBVHNode nodes[];
};

layout(std430, binding = 1) readonly buffer Primitives {
    int primitives[];
};

layout(std430, binding = 2) readonly buffer Spheres {
    Sphere spheres[];
};
```

---

## ✅ 数据结构（CS端定义）

```glsl
struct PackedBVHNode {
    vec4 aabbMin;  // xyz: min, w: left
    vec4 aabbMax;  // xyz: max, w: right
    ivec4 meta;    // x: primitiveOffset, y: primitiveCount
};

struct Sphere {
    vec3 center;
    float radius;
};

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Hit {
    float t;
    int primId;
};
```

---

## ✅ Ray-AABB 相交（Slab 法）

```glsl
bool intersectAABB(Ray ray, vec3 minB, vec3 maxB) {
    vec3 invD = 1.0 / ray.dir;
    vec3 t0 = (minB - ray.origin) * invD;
    vec3 t1 = (maxB - ray.origin) * invD;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar = min(min(tmax.x, tmax.y), tmax.z);
    return tNear <= tFar && tFar >= 0.0;
}
```

---

## ✅ Ray-Sphere 相交

```glsl
bool intersectSphere(Ray ray, Sphere s, out float tHit) {
    vec3 oc = ray.origin - s.center;
    float b = dot(oc, ray.dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    tHit = -b - h;
    if (tHit < 0.0) tHit = -b + h;
    return tHit >= 0.0;
}
```

---

## ✅ 核心函数：traverseBVH

```glsl
bool traverseBVH(int nodeIdx, Ray ray, out Hit hit) {
    bool found = false;
    hit.t = 1e30;
    hit.primId = -1;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = nodeIdx;

    while (stackPtr > 0) {
        int idx = stack[--stackPtr];
        PackedBVHNode node = nodes[idx];

        if (!intersectAABB(ray, node.aabbMin.xyz, node.aabbMax.xyz))
            continue;

        if (node.meta.y > 0) {
            // Leaf node
            for (int i = 0; i < node.meta.y; ++i) {
                int primIdx = primitives[node.meta.x + i];
                Sphere s = spheres[primIdx];
                float t;
                if (intersectSphere(ray, s, t)) {
                    if (t < hit.t) {
                        hit.t = t;
                        hit.primId = primIdx;
                        found = true;
                    }
                }
            }
        } else {
            // Inner node — push children to stack
            stack[stackPtr++] = int(node.aabbMin.w);  // left
            stack[stackPtr++] = int(node.aabbMax.w);  // right
        }
    }

    return found;
}
```

---

## ✅ 示例用法（在 `main()` 中）

```glsl
Ray ray = makeRayForPixel(gl_GlobalInvocationID.xy); // 你自己定义
Hit hit;
bool hitFound = traverseBVH(0, ray, hit);

if (hitFound) {
    // 输出颜色、ID、法线等
    imageStore(outputImage, ivec2(gl_GlobalInvocationID.xy), vec4(1.0, 0.5, 0.0, 1.0));
} else {
    imageStore(outputImage, ivec2(gl_GlobalInvocationID.xy), vec4(0.0));
}
```

---

## 🧠 总结

* 每条 ray 调用 `traverseBVH()`；
* 内部 stack-based 遍历，适合 compute shader；
* leaf 节点才访问 `primitives[]` 和 `spheres[]` 做精确 hit test；
* 最终输出结果由你定义的 image / buffer 写出。

---

 */
#endif

