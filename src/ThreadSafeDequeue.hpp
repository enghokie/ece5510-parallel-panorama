#pragma once

#include <deque>
#include <atomic>
#include <mutex>
#include <condition_variable>

template <class T> class ThreadSafeDequeue
{
public:
    ThreadSafeDequeue()
        : _numElements(0)
        , _quit(false)
    {}

    ~ThreadSafeDequeue()
    {
        stop();
    }

    void push(T& t)
    {
        bool mustWakeDequeuers(false);
        std::lock_guard<std::mutex> lock(_enqueueLock);
        _queue.push_back(std::move(t));
        
        // Notify dequeuers if the queue was empty
        bool notifyDequeuers = _numElements.fetch_add(1) > 0 ? false : true;
        if (notifyDequeuers)
            _notEmptyCondition.notify_all();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(_dequeueLock);
        T val;
        int numElements(0);
        while (!_quit.load(std::memory_order_relaxed))
        {
            numElements = _numElements.load(std::memory_order_relaxed);
            while (numElements == 0 && !_quit.load(std::memory_order_relaxed))
            {
                _notEmptyCondition.wait(lock);
                numElements = _numElements.load(std::memory_order_relaxed);
            }
            if (_numElements.compare_exchange_weak(numElements, numElements - 1))
            {
                val = std::move(_queue.front());
                _queue.pop_front();
                return val;
            }
        }
        return val;
    }

    void stop()
    {
        _quit.store(true, std::memory_order_relaxed);
        _notEmptyCondition.notify_all();
    }

private:
    std::deque<T> _queue;
    mutable std::mutex _enqueueLock;
    mutable std::mutex _dequeueLock;
    std::condition_variable _notEmptyCondition;
    std::atomic_bool _quit;
    std::atomic_int _numElements;
};

