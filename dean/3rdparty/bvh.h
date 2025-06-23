#ifndef BVH_H
#define BVH_H

#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

#include "utils.h"

#define BUILD_BVH_LOG 0  // todo, use a nicer way to print log.

#define LAMBERTIAN 0
#define METAL 1
#define DIELECTRIC 2

inline float random_float() {
    // Returns a random real in [0,1).
    return std::rand() / (RAND_MAX + 1.0);
}

inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

inline glm::vec3 randomVec3() {
    return glm::vec3(random_float(), random_float(), random_float());
}

inline glm::vec3 random(float min, float max) {
    return glm::vec3(random_float(min, max), random_float(min, max), random_float(min, max));
}

/*

todo,
把rt in one weekend的随机生成ball的代码port过来, 然后做成bvh, 打印fps, 看看O(n)和O(logn)的区别
    done
在cs里面用rt in one weekend的方式生成ray.
    done
不同材质
    done
bounce多次
增加 light 和阴影

*/
struct Sphere {
    glm::vec3 center;
    float radius;

    // material
    glm::vec3 color;
    int materialType;  // LAMBERTIAN, METAL, DIELECTRIC

    float fuzz;
    float refractionIndex;
    float pad1;
    float pad2;
};

class Scene {
public:
    const std::vector<Sphere>& getSpheres() const {
        return spheres;
    }

    void generate(bool useExample = false) {
        std::srand(42);  // 固定种子，保持一致

        if (useExample) {
            // 在z上排列, 无序
            float r = 0.2f;
            spheres = {
                // {center,                   radius, color,                              materialType,fuzz,refractionIndex}
                {glm::vec3(0.0f, 0.0f, 2.5f), r, glm::vec3(0.380012, 0.506085, 0.762437), LAMBERTIAN, 1.0, 1.0, 0, 0},
                {glm::vec3(0.0f, 0.0f, 1.5f), r, glm::vec3(0.596282, 0.140784, 0.017972), LAMBERTIAN, 1.0, 1.0, 0, 0},
                {glm::vec3(0.0f, 0.0f, 6.0f), r, glm::vec3(0.288507, 0.465652, 0.665070), LAMBERTIAN, 1.0, 1.0, 0, 0},
                {glm::vec3(0.0f, 0.0f, 1.0f), r, glm::vec3(0.101047, 0.293493, 0.813446), LAMBERTIAN, 1.0, 1.0, 0, 0},
                {glm::vec3(0.0f, 0.0f, 3.0f), r, glm::vec3(0.365924, 0.221622, 0.058332), LAMBERTIAN, 1.0, 1.0, 0, 0},
                {glm::vec3(0.0f, 0.0f, 2.0f), r, glm::vec3(0.051231, 0.430547, 0.454086), LAMBERTIAN, 1.0, 1.0, 0, 0},
            };
            return;
        }

        // 地面是一个巨大球体
        spheres.push_back({glm::vec3(0, -1000, 0), 1000, glm::vec3(0.5, 0.5, 0.5), LAMBERTIAN});

        // 很多小球
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = random_float();
                // 随机生成球的中心位置.
                glm::vec3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

                if ((center - glm::vec3(4, 0.2, 0)).length() > 0.9) {
                    //     shared_ptr<material> sphere_material;

                    if (choose_mat < 0.8) {
                        // diffuse
                        glm::vec3 color = randomVec3() * randomVec3();
                        spheres.push_back({center, 0.2, color, LAMBERTIAN});
                    } else if (choose_mat < 0.95) {
                        // metal
                        glm::vec3 color = random(0.5, 1);
                        float fuzz = random_float(0, 0.5);
                        spheres.push_back({center, 0.2, color, METAL, fuzz});
                    } else {
                        // glass
                        float refraction_index = 1.5;
                        spheres.push_back({center, 0.2, glm::vec3(0.0f), DIELECTRIC, 0, refraction_index});
                    }
                }
            }
        }

        // 3个大球
        spheres.push_back({glm::vec3(0, 1, 0), 1.0, glm::vec3(0.0f), DIELECTRIC, 0, 1.5});
        spheres.push_back({glm::vec3(-4, 1, 0), 1.0, glm::vec3(0.4, 0.2, 0.1), LAMBERTIAN});
        spheres.push_back({glm::vec3(4, 1, 0), 1.0, glm::vec3(0.7, 0.6, 0.5), METAL, 0.0});

        PRINTF("sphere size=%ld\n", spheres.size());
    }

private:
    std::vector<Sphere> spheres;
};

// bvh array里面的第一个肯定是root node.
// 如果是叶子 (primitiveCount > 0),
//     当前AABB包含了primitiveCount个primitive, 从 primitives[primitiveOffset]开始
// 如果是内部节点 (primitiveCount == 0),
//     左子节点为bvhNodes[leftChild], 右子节点为bvhNodes[rightChild]
struct PackedBVHNode {
    // xyz: aabb.min
    // w: leftChild, 对于内部节点，表示左子节点索引；对于叶子节点，未使用
    glm::vec4 aabbMin;
    // xyz: aabb.max
    // w: rightChild, 对于内部节点，表示右子节点索引；对于叶子节点，未使用
    glm::vec4 aabbMax;
    // x: primitiveOffset, 对于叶子节点，表示 primitives[] 的起始位置；对于内部节点，未使用
    // y: primitiveCount, == 0 表示内部节点；> 0 表示叶子节点，表示 primitive 个数
    // z: unused, w: unused
    glm::ivec4 meta;
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
    BVH() {}
    ~BVH() {}
    void generate(const std::vector<Sphere>& spheres) {
        // 测试, 在运行到的任意地方用: BVH bvh;

        primitives.resize(spheres.size());
        std::iota(primitives.begin(), primitives.end(), 0);
        bvhNodes.clear();
        int rootIndex = buildBVHRecursive(0, primitives.size(), spheres);

        if (BUILD_BVH_LOG) {
            std::cout << "[BUILD DONE] root index = " << rootIndex << std::endl;
            printBVHArray();
        }

        /*
primitive array:3 1 5 0 4 2

[BUILD DONE] root index = 0
node 0 [INNER] left=1 right=6
node 1 [INNER] left=2 right=3
node 2 [LEAF]  offset=0 count=2
node 3 [INNER] left=4 right=5
node 4 [LEAF]  offset=2 count=1
node 5 [LEAF]  offset=3 count=2
node 6 [LEAF]  offset=5 count=1

           0
         /   \
        1     6
      /  \
     2    3
         / \
        4   5

        */
    }

    const std::vector<int>& getPrimitives() const {
        return primitives;
    }

    const std::vector<PackedBVHNode>& getBVHNodes() const {
        return bvhNodes;
    }

private:
    // primitives 是一个数组，里面是 球array spheres的索引
    std::vector<int> primitives;
    std::vector<PackedBVHNode> bvhNodes;

    void printPrimitiveArray() {
        for (size_t i = 0; i < primitives.size(); ++i) {
            std::cout << primitives[i] << " ";
        }
        std::cout << std::endl;
    }

    inline bool isLeaf(const PackedBVHNode& node) {
        return node.meta.y > 0;
    }

    void printBVHArray() {
        for (size_t i = 0; i < bvhNodes.size(); ++i) {
            const auto& node = bvhNodes[i];
            if (isLeaf(node)) {
                std::cout << "node " << i << " [LEAF]  offset=" << node.meta.x
                          << " count=" << node.meta.y << "\n";
            } else {
                std::cout << "node " << i << " [INNER] left=" << node.aabbMin.w
                          << " right=" << node.aabbMax.w << "\n";
            }
        }
    }

    // 输入: spheres array, primitives array
    // 输出: bvhNodes
    // 返回: bvh array的当前构建的node的index.
    //
    // 这个函数就是对primitives array里面startIndex开始的count个球做:
    // 1. 计算整个包围count个球的AABB
    // 2, 得到这个AABB的最长轴axis, x/0, y/1, z/2, 即在哪个轴上分布的最广, 并得到最长轴的中点splitPos
    // 3, 用splitPos对primitives array做一次partition, 在最长轴上, 球的位置比 splitPos 小的放左边, 大的放右边.
    //     注意这里用户data spheres array不会动, 改变的是primitives array里的值, 也就是指向spheres array里面的球索引.
    // 4, 递归调用 buildBVHRecursive 直到count<=2, 结束
    int buildBVHRecursive(int startIndex, int count, const std::vector<Sphere>& spheres) {
        if (BUILD_BVH_LOG) {
            std::cout << "[ENTER] buildBVHRecursive(start=" << startIndex
                      << ", count=" << count << "), bvhNodes.size()=" << bvhNodes.size() << std::endl;
        }
        // 计算count个球的AABB
        AABB bounds;
        for (int i = startIndex; i < startIndex + count; ++i) {
            bounds.expand(computeSphereAABB(spheres[primitives[i]]));
        }

        int currentIndex = static_cast<int>(bvhNodes.size());
        bvhNodes.emplace_back();
        if (BUILD_BVH_LOG) {
            std::cout << "[NEW NODE] index=" << currentIndex << std::endl;
        }

        bvhNodes[currentIndex].aabbMin = glm::vec4(bounds.min, -1.0f);
        bvhNodes[currentIndex].aabbMax = glm::vec4(bounds.max, -1.0f);

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
            // 创建叶子节点
            bvhNodes[currentIndex].meta.y = count;
            bvhNodes[currentIndex].meta.x = startIndex;
            return currentIndex;
        }

        // 内部节点
        bvhNodes[currentIndex].meta.y = 0;

        // 选择分裂轴：最长轴
        glm::vec3 extent = bounds.max - bounds.min;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        // 根据质心分裂, 就是AABB里面最长轴的中点.
        float splitPos = bounds.centroid()[axis];

        // 指定primitives的开始位置和结束位置half-open. 即把primitives过一遍, 按splitPos分成两半.
        // 小的在左边, 大的在右边.
        // 注意这个partition会把primitives里的值改掉, 返回的是迭代器.
        auto mid = std::partition(
            primitives.begin() + startIndex,
            primitives.begin() + startIndex + count,
            [&](int idx) {  // idx为primitives array里面的值, 不是地址.
                // 计算每个球的AABB
                AABB aabb = computeSphereAABB(spheres[idx]);
                // 如果球的质心在分裂轴的左边，就返回true
                return aabb.centroid()[axis] < splitPos;
            });

        int midIndex = static_cast<int>(mid - primitives.begin());

        if (BUILD_BVH_LOG) {
            std::cout << "partition axis = " << axis << ", splitPos = " << splitPos
                      << ", midIndex in primitives array = " << midIndex
                      << ", start = " << startIndex
                      << ", end = " << (startIndex + count) << std::endl;
        }

        // 如果分裂失败，强制一半一半
        if (midIndex == startIndex || midIndex == startIndex + count) {
            std::cout << "[PARTITION FAIL] fallback triggered, force split at "
                      << (startIndex + count / 2) << std::endl;
            // fallback: 强制切
            midIndex = startIndex + count / 2;
            std::rotate(primitives.begin() + startIndex,
                        primitives.begin() + midIndex,
                        primitives.begin() + startIndex + count);
        }

        if (BUILD_BVH_LOG) {
            // print primitive array after partition.
            printPrimitiveArray();

            // 递归构建左右子树
            std::cout << "[RECURSE] left=[" << startIndex << ", " << (midIndex - startIndex) << "], "
                      << "right=[" << midIndex << ", " << (startIndex + count - midIndex) << "]"
                      << std::endl;
        }
        int leftChild = buildBVHRecursive(startIndex, midIndex - startIndex, spheres);
        int rightChild = buildBVHRecursive(midIndex, startIndex + count - midIndex, spheres);

        bvhNodes[currentIndex].aabbMin.w = leftChild;
        bvhNodes[currentIndex].aabbMax.w = rightChild;

        if (BUILD_BVH_LOG) {
            std::cout << "<< return nodeIndex = " << currentIndex << std::endl;
        }
        assert(startIndex < midIndex && midIndex < startIndex + count && "midIndex fallback failed");
        assert(leftChild != rightChild && "left and right child identical");

        return currentIndex;
    }
};

#endif
