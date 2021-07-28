#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

#include <quiver/functor.cu.hpp>
#include <quiver/trace.hpp>

constexpr static const int BLOCK_SIZE = 256;
constexpr static const size_t TILE_SIZE = 1024;

template <typename IdType>
class HostOrderedHashTable;

template <typename IdType>
class DeviceOrderedHashTable
{
  public:
    friend class HostOrderedHashTable<IdType>;
    static constexpr IdType kEmptyKey = static_cast<IdType>(-1);
    /**
     * \brief An entry in the hashtable.
     */
    struct Mapping {
        /**
         * \brief The ID of the item inserted.
         */
        IdType key;
        /**
         * \brief The index of the item when inserted into the hashtable (e.g.,
         * the index within the array passed into FillWithDuplicates()).
         */
        int index;
        /**
         * \brief The index of the item in the unique list.
         */
        int local;
    };

    typedef const Mapping *ConstIterator;
    typedef Mapping *Iterator;

    DeviceOrderedHashTable(Mapping *table = nullptr, size_t size = 0)
        : table_(table), size_(size)
    {
    }

    /**
     * \brief Find the non-mutable mapping of a given key within the hash table.
     *
     * WARNING: The key must exist within the hashtable. Searching for a key not
     * in the hashtable is undefined behavior.
     *
     * \param id The key to search for.
     *
     * \return An iterator to the mapping.
     */
    inline __device__ Iterator Search(const IdType id)
    {
        const size_t pos = SearchForPosition(id);

        return &table_[pos];
    }

    /**
     * @brief Insert key-index pair into the hashtable.
     *
     * @param id The ID to insert.
     * @param index The index at which the ID occured.
     *
     * @return An iterator to inserted mapping.
     */
    inline __device__ Iterator Insert(const IdType id, const size_t index)
    {
        size_t pos = Hash(id);

        // linearly scan for an empty slot or matching entry
        IdType delta = 1;
        while (!AttemptInsertAt(pos, id, index)) {
            pos = Hash(pos + delta);
            delta += 1;
        }

        return GetMutable(pos);
    }

  protected:
    Mapping *table_;
    size_t size_;

    /**
     * \brief Search for an item in the hash table which is known to exist.
     *
     * WARNING: If the ID searched for does not exist within the hashtable, this
     * function will never return.
     *
     * \param id The ID of the item to search for.
     *
     * \return The the position of the item in the hashtable.
     */
    inline __device__ size_t SearchForPosition(const IdType id) const
    {
        IdType pos = Hash(id);

        // linearly scan for matching entry
        IdType delta = 1;
        while (table_[pos].key != id) {
            pos = Hash(pos + delta);
            delta += 1;
        }

        return pos;
    }

    inline __device__ bool AttemptInsertAt(const size_t pos, const IdType id,
                                           const size_t index)
    {
        using Type = unsigned long long int;
        const IdType key =
            atomicCAS(reinterpret_cast<Type *>(&GetMutable(pos)->key),
                      static_cast<Type>(kEmptyKey), static_cast<Type>(id));
        if (key == kEmptyKey || key == id) {
            // we either set a match key, or found a matching key, so then place
            // the minimum index in position. Match the type of atomicMin, so
            // ignore linting
            atomicMin(
                reinterpret_cast<Type *>(&GetMutable(pos)->index),  // NOLINT
                static_cast<Type>(index));                          // NOLINT
            return true;
        } else {
            // we need to search elsewhere
            return false;
        }
    }

    inline __device__ Mapping *GetMutable(const size_t pos)
    {
        assert(pos < this->size_);
        // The parent class Device is read-only, but we ensure this can only be
        // constructed from a mutable version of OrderedHashTable, making this
        // a safe cast to perform.
        return this->table_ + pos;
    }

    /**
     * \brief Hash an ID to a to a position in the hash table.
     *
     * \param id The ID to hash.
     *
     * \return The hash.
     */
    inline __device__ size_t Hash(const IdType id) const { return id % size_; }
};

template <typename IdType>
class HostOrderedHashTable
{
  public:
    using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;
    static constexpr IdType kEmptyKey = static_cast<IdType>(-1);
    // Must be uniform bytes for memset to work
    HostOrderedHashTable(size_t num, int scale)
    {
        const size_t next_pow2 =
            1 << static_cast<size_t>(1 + std::log2(num >> 1));
        auto size = next_pow2 << scale;
        void *p;
        cudaMalloc(&p, size * sizeof(Mapping));
        cudaMemset(p, kEmptyKey, size * sizeof(Mapping));
        device_table_ = DeviceOrderedHashTable<IdType>(
            reinterpret_cast<Mapping *>(p), size);
    }
    ~HostOrderedHashTable() { cudaFree(device_table_.table_); }
    DeviceOrderedHashTable<IdType> DeviceHandle() { return device_table_; }

  private:
    DeviceOrderedHashTable<IdType> device_table_;
};

template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(const IdType *const items,
                                        const int64_t num_items,
                                        DeviceOrderedHashTable<IdType> table)
{
    assert(BLOCK_SIZE == blockDim.x);

    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

    using Iterator = typename DeviceOrderedHashTable<IdType>::Iterator;

#pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end;
         index += BLOCK_SIZE) {
        if (index < num_items) {
            Iterator pos = table.Insert(items[index], index);

            // since we are only inserting unique items, we know their local id
            // will be equal to their index
            pos->local = static_cast<IdType>(index);
        }
    }
}

template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
generate_hashmap_duplicates(const IdType *const items, const int64_t num_items,
                            DeviceOrderedHashTable<IdType> table)
{
    assert(BLOCK_SIZE == blockDim.x);

    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end;
         index += BLOCK_SIZE) {
        if (index < num_items) { table.Insert(items[index], index); }
    }
}

template <typename T>
class reorder_functor
{
    const T *output_sum;
    const T *counts;
    const T *values;
    T *output;

  public:
    reorder_functor(const T *output_sum, const T *counts, const T *values,
                    T *output)
        : output_sum(output_sum), counts(counts), values(values), output(output)
    {
    }

    __device__ void operator()(const thrust::tuple<T, T> &t)
    {
        const size_t pos = thrust::get<0>(t);
        size_t beg = thrust::get<1>(t);
        size_t output_beg = output_sum[pos];
        const size_t count = counts[pos];
        for (int j = 0; j < count; j++) {
            output[output_beg++] = values[beg++];
        }
    }
};

template <typename T>
void reorder_output(const thrust::device_vector<T> &p1,
                    const thrust::device_vector<T> &p2,
                    const thrust::device_vector<T> &order,
                    const thrust::device_vector<T> &c,
                    const thrust::device_vector<T> &v,
                    thrust::device_vector<T> &out, cudaStream_t stream)
{
    const auto policy = thrust::cuda::par.on(stream);
    auto beg = thrust::make_zip_iterator(
        thrust::make_tuple(order.begin(), p1.begin()));
    auto end =
        thrust::make_zip_iterator(thrust::make_tuple(order.end(), p1.end()));
    thrust::for_each(policy, beg, end,
                     reorder_functor<T>(thrust::raw_pointer_cast(p2.data()),
                                        thrust::raw_pointer_cast(c.data()),
                                        thrust::raw_pointer_cast(v.data()),
                                        thrust::raw_pointer_cast(out.data())));
}

// Given a partial permutation p of {0, ..., n - 1} with only the first m values
// known, convert p into a full permutation.
// this implementation outputs the smallest one in lexicographical order.
template <typename T>
void complete_permutation(thrust::device_vector<T> &p, size_t n,
                          cudaStream_t stream)
{
    using it = thrust::counting_iterator<T>;
    const size_t m = p.size();
    const auto policy = thrust::cuda::par.on(stream);
    thrust::device_vector<thrust::pair<T, T>> q(n);
    thrust::for_each(policy, it(0), it(n),
                     [q = thrust::raw_pointer_cast(q.data()),
                      p = thrust::raw_pointer_cast(p.data()),
                      m = m]  //
                     __device__(T i) {
                         q[i].first = m;
                         q[i].second = i;
                     });
    thrust::for_each(policy, it(0), it(m),
                     [q = thrust::raw_pointer_cast(q.data()),
                      p = thrust::raw_pointer_cast(p.data())]  //
                     __device__(T i) { q[p[i]].first = i; });
    thrust::sort(policy, q.begin(), q.end());
    p.resize(n);
    thrust::transform(policy, q.begin(), q.end(), p.begin(), thrust_get<1>());
}

// given a permutation q of {0, ..., n - 1}, find q, the inverse of p, such that
// q[p[i]] == i for i in {0, ..., n - 1}
template <typename T>
void inverse_permutation(const thrust::device_vector<T> &p,
                         thrust::device_vector<T> &q, cudaStream_t stream)
{
    const size_t n = p.size();
    q.resize(n);
    using it = thrust::counting_iterator<T>;
    thrust::for_each(thrust::cuda::par.on(stream), it(0), it(n),
                     [p = thrust::raw_pointer_cast(p.data()),
                      q = thrust::raw_pointer_cast(q.data())]  //
                     __device__(T i) { q[p[i]] = i; });
}

template <typename T>
void permute_value(const thrust::device_vector<T> &p,
                   thrust::device_vector<T> &a, cudaStream_t stream)
{
    const auto policy = thrust::cuda::par.on(stream);
    thrust::transform(policy, a.begin(), a.end(), a.begin(),
                      value_at<T>(thrust::raw_pointer_cast(p.data())));
}

template <typename T>
thrust::device_vector<T> permute(const thrust::device_vector<T> &p,
                                 const thrust::device_vector<T> &a,
                                 cudaStream_t stream)
{
    const size_t n = a.size();
    thrust::device_vector<T> b(n);
    using it = thrust::counting_iterator<T>;
    thrust::for_each(thrust::cuda::par.on(stream), it(0), it(n),
                     [p = thrust::raw_pointer_cast(p.data()),
                      a = thrust::raw_pointer_cast(a.data()),
                      b = thrust::raw_pointer_cast(b.data())]  //
                     __device__(T i) { b[i] = a[p[i]]; });
    return b;
}

template <typename T>
void _reindex_with(const thrust::device_vector<T> &a,
                   const thrust::device_vector<T> &I,
                   thrust::device_vector<T> &c)
{
    c.resize(a.size());
    thrust::lower_bound(I.begin(), I.end(), a.begin(), a.end(), c.begin());
}

template <typename T, typename P>
void _reindex_with(const P &policy, const thrust::device_vector<T> &a,
                   const thrust::device_vector<T> &I,
                   thrust::device_vector<T> &c)
{
    c.resize(a.size());
    thrust::lower_bound(policy, I.begin(), I.end(), a.begin(), a.end(),
                        c.begin());
}

template <typename T>
void reindex(const thrust::device_vector<T> &a, thrust::device_vector<T> &b,
             thrust::device_vector<T> &c)
{
    {
        b.resize(a.size());
        thrust::copy(a.begin(), a.end(), b.begin());
        thrust::sort(b.begin(), b.end());
        const size_t m = thrust::unique(b.begin(), b.end()) - b.begin();
        b.resize(m);
    }
    _reindex_with(a, b, c);
}

template <typename T>
void reindex_with_seeds(const thrust::device_vector<T> &a,
                        const thrust::device_vector<T> &s,
                        thrust::device_vector<T> &b,
                        thrust::device_vector<T> &c)
{
    TRACE_SCOPE("reindex_with_seeds<thrust>");

    // reindex(a, b, c);
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::sort unique");

        b.resize(a.size() + s.size());
        thrust::copy(a.begin(), a.end(), b.begin());
        thrust::copy(s.begin(), s.end(), b.begin() + a.size());
        thrust::sort(b.begin(), b.end());
        const size_t m = thrust::unique(b.begin(), b.end()) - b.begin();
        b.resize(m);
    }
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::_reindex_with 1");
        _reindex_with(a, b, c);
    }

    thrust::device_vector<T> s1;
    s1.reserve(b.size());
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::_reindex_with 2");
        _reindex_with(s, b, s1);
    }
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::complete_permutation");
        complete_permutation(s1, b.size());
    }

    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::permute");
        b = permute(s1, b);
    }

    thrust::device_vector<T> s2;
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::inverse_permutation");
        inverse_permutation(s1, s2);
    }
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::permute_value");
        permute_value(s2, c);
    }
}

template <typename T>
void reindex(const std::vector<T> &a, std::vector<T> &b, std::vector<T> &c)
{
    thrust::device_vector<T> cuda_a(a.size());
    thrust::copy(a.begin(), a.end(), cuda_a.begin());

    thrust::device_vector<T> cuda_b;
    thrust::device_vector<T> cuda_c;
    reindex(cuda_a, cuda_b, cuda_c);

    b.resize(cuda_b.size());
    thrust::copy(cuda_b.begin(), cuda_b.end(), b.begin());
    c.resize(cuda_c.size());
    thrust::copy(cuda_c.begin(), cuda_c.end(), c.begin());
}

template <typename T>
void reindex_with_seeds(const std::vector<T> &a, const std::vector<T> &s,
                        std::vector<T> &b, std::vector<T> &c)
{
    thrust::device_vector<T> cuda_a(a.size());
    thrust::device_vector<T> cuda_s(s.size());
    {
        TRACE_SCOPE("reindex_with_seeds::copy1");
        thrust::copy(a.begin(), a.end(), cuda_a.begin());
        thrust::copy(s.begin(), s.end(), cuda_s.begin());
    }

    thrust::device_vector<T> cuda_b;
    thrust::device_vector<T> cuda_c;
    reindex_with_seeds(cuda_a, cuda_s, cuda_b, cuda_c);

    {
        TRACE_SCOPE("reindex_with_seeds::copy2");
        b.resize(cuda_b.size());
        thrust::copy(cuda_b.begin(), cuda_b.end(), b.begin());
        c.resize(cuda_c.size());
        thrust::copy(cuda_c.begin(), cuda_c.end(), c.begin());
    }
}

template <typename T>
void reindex_with_seeds(size_t l, const T *s, size_t r, const T *a,
                        std::vector<T> &b, std::vector<T> &c)
{
    thrust::device_vector<T> cuda_a(r);
    thrust::device_vector<T> cuda_s(l);
    {
        TRACE_SCOPE("reindex_with_seeds::copy1");

        thrust::copy(a, a + r, cuda_a.begin());
        thrust::copy(s, s + l, cuda_s.begin());
    }

    thrust::device_vector<T> cuda_b;
    thrust::device_vector<T> cuda_c;
    reindex_with_seeds(cuda_a, cuda_s, cuda_b, cuda_c);

    {
        TRACE_SCOPE("reindex_with_seeds::copy2");
        b.resize(cuda_b.size());
        thrust::copy(cuda_b.begin(), cuda_b.end(), b.begin());
        c.resize(cuda_c.size());
        thrust::copy(cuda_c.begin(), cuda_c.end(), c.begin());
    }
}
