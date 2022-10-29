# Task https://leetcode.com/problems/find-servers-that-handled-most-number-of-requests/
import heapq
# initial data
k = 3
arrival = [2,6,7,8] #[1,2,3,4,5] #[1,2,3,4] #
load = [1,3,1,4] #[5,2,3,3,3] #[1,2,1,2] #

# solution that works.
# This type of solution is correct and has complexity of O(nlogn) in case of a solution type
# This is better than naive solution is case of algorithm itself.
class Solution:
    def busiestServers(self, k, arrival, load):

        available = list(range(k)) # already a min-heap
        busy = []
        res = [0] * k
        for i, a in enumerate(arrival):
            while busy and busy[0][0] <= a: # these are done, put them back as available
                _, x = heapq.heappop(busy)
                heapq.heappush(available, i + (x-i)%k) # invariant: min(available) is at least i, at most i+k-1
            if available:
                j = heapq.heappop(available) % k
                heapq.heappush(busy, (a+load[i],j))
                res[j] += 1
        a = max(res)
        return [i for i in range(k) if res[i] == a]

# In the case of a business problem, this solution is not the best.
# A process in the above code that searches for a free server and requests a connection to it
# using an algorithm, even if the previous
# one has completed the process. This solution requires more servers and, as a result, costs more.

# more business efficient solution
class Solution:
    def busiestServers_n(self, k, arrival, load):
        servers = {}
        for server in range(k):
            servers[server] = 0

        servers_cnt = servers.copy()

        for i in range(len(arrival)):

                servers = dict(sorted(servers.items(), key=lambda item: item[1], reverse=True))
                # print(servers)
                for s in servers:
                    if servers[s] != 0:
                        servers[s] -= 1
                    if servers[s] == 0:
                        servers[s] = load[i]
                        servers_cnt[s] += 1
                        break

        ser = []
        max_v = max(servers_cnt.items(), key=lambda item: max(item))[1]
        for i in servers_cnt:
            if servers_cnt[i] >= max_v:
                ser.append(i)
                max_v = servers_cnt[i]

        return ser

# but is needs some refactoring to become algorithm efficient as well.
# Last algorithm make servers work harder and feed new request to the server straight after
# it finished previous one. (only if there is a request).
# Need some tests to check more timelines and 'what if' cases with more complex partial timelines (as in last case).
