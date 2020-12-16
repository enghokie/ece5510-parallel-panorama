/***
ParallelPanorama: Concurrently stitches together images from files and displays them.
Copyright (C) 2020 Braedon Dickerson and Amir Kimiyaie
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software Foundation,
   Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
***/

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

