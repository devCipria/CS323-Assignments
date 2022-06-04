# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def longest_palindrome_subseq(s):
    """
        :type s: str
        :rtype: int
        """
    n = len(s)
    dp = [1] * n
    for j in range(1, len(s)):
        pre = dp[j]
        for i in reversed(range(0, j)):
            tmp = dp[i]
            if s[i] == s[j]:
                dp[i] = 2 + pre if i + 1 <= j - 1 else 2
            else:
                dp[i] = max(dp[i + 1], dp[i])
            pre = tmp
    return dp[0]


def main():
    print(longest_palindrome_subseq("queenscollegofcuny"))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
