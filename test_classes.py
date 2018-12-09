"""Base classes for tests.

Champlain College CSI-480, Fall 2018
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""


from util import raise_not_defined


class Question:
    """Class which models a question in a project.

    Note that questions have a maximum number of points they are worth,
    and are composed of a series of test cases.
    """

    def __init__(self, question_dict, display):
        """Create Question instance given dictionary and display object."""
        self.max_points = int(question_dict['max_points'])
        self.test_cases = []
        self.display = display

    def add_test_case(self, test_case, thunk):
        """Add a test case.

        Note that 'thunk' must be a function which accepts a single argument,
        namely a 'grading' object
        """
        self.test_cases.append((test_case, thunk))

    def execute(self, grades):
        """Run the test and puts the result in grades.

        This will raise an error if not overridden.
        """
        raise_not_defined()


class PassAllTestsQuestion(Question):
    """Question requiring all tests be passed in order to receive credit."""

    def execute(self, grades):
        """Run the tests and put result in the grades object.

        Overrides Question.execute
        """
        self.tests_failed = False
        grades.assign_zero_credit()
        for _, f in self.test_cases:
            if not f(grades):
                self.tests_failed = True
        if self.tests_failed:
            grades.fail("Tests failed.")
        else:
            grades.assign_full_credit()


class ExtraCreditPassAllTestsQuestion(PassAllTestsQuestion):
    """Class extending PassAllTestsQuestion with extra credit."""

    def __init__(self, question_dict, display):
        """Extend Question.__init__ to get extra_points out of the dict."""
        super().__init__(question_dict, display)
        self.extra_points = int(question_dict['extra_points'])

    def execute(self, grades):
        """Extend PassAllTestsQuestion.execute to add extra credit."""
        super().execute(grades)
        if not self.tests_failed:
            grades.add_points(self.extra_points)


class HackedPartialCreditQuestion(Question):
    """Question in which partial credit is given for some test cases.

    Test cases with a ``points'' property can receive partial credit.
    All other tests are mandatory and must be passed.
    """

    def execute(self, grades):
        """Run the tests and put result in the grades object.

        Overrides Question.execute
        """
        grades.assign_zero_credit()

        points = 0
        passed = True
        for test_case, f in self.test_cases:
            test_result = f(grades)
            if "points" in test_case.test_dict:
                if test_result:
                    points += float(test_case.test_dict["points"])
            else:
                passed = passed and test_result

        # FIXME: Below terrible hack to match q3's logic
        if int(points) == self.max_points and not passed:
            grades.assign_zero_credit()
        else:
            grades.add_points(int(points))


class Q6PartialCreditQuestion(Question):
    """Fails any test which returns False.

    Otherwise doesn't effect the grades object.
    Partial credit tests will add the required points.
    """

    def execute(self, grades):
        """Run the tests and put result in the grades object.

        Overrides Question.execute
        """
        grades.assign_zero_credit()

        results = []
        for _, f in self.test_cases:
            results.append(f(grades))
        if False in results:
            grades.assign_zero_credit()


class PartialCreditQuestion(Question):
    """Fails any test which returns False.

    Otherwise doesn't effect the grades object.
    Partial credit tests will add the required points.
    """

    def execute(self, grades):
        """Run the tests and put result in the grades object.

        Overrides Question.execute
        """
        grades.assign_zero_credit()

        for _, f in self.test_cases:
            if not f(grades):
                grades.assign_zero_credit()
                grades.fail("Tests failed.")
                return False


class NumberPassedQuestion(Question):
    """Grade is the number of test cases passed."""

    def execute(self, grades):
        """Run the test and put result in the grades object.

        Overrides Question.execute
        """
        points_per_test = self.max_points / len(self.test_cases)
        grades.add_points([f(grades) for _, f in self.test_cases].count(True)
                          * points_per_test)


class TestCase:
    """Template modeling a generic test case."""

    def __init__(self, question, test_dict):
        """Create a generic test case."""
        self.question = question
        self.test_dict = test_dict
        self.path = test_dict['path']
        self.messages = []

    def get_path(self):
        """Return the path."""
        return self.path

    def __str__(self):
        """Return a string of this tests case.

        This will raise an error if not overridden.
        """
        raise_not_defined()

    def execute(self, grades, module_dict, solution_dict):
        """Run the test.

        This will raise an error if not overridden.
        """
        raise_not_defined()

    def write_solution(self, module_dict, file_path):
        """Write solution for the test.

        This will raise an error if not overridden.
        """
        raise_not_defined()

    # Tests should call the following messages for grading
    # to ensure a uniform format for test output.
    #
    # TODO: this is hairy, but we need to fix grading.py's interface
    # to get a nice hierarchical project - question - test structure,
    # then these should be moved into Question proper.
    def test_pass(self, grades):
        """Add messages for test passing."""
        grades.add_message('PASS: %s' % (self.path, ))
        for line in self.messages:
            grades.add_message('    %s' % (line, ))
        return True

    def test_fail(self, grades):
        """Add messages for test failing."""
        grades.add_message('FAIL: %s' % (self.path, ))
        for line in self.messages:
            grades.add_message('    %s' % (line, ))
        return False

    # This should really be question level?
    def test_partial(self, grades, points, max_points):
        """Add messages for partial credit."""
        grades.add_points(points)
        extra_credit = max(0, points - max_points)
        regular_credit = points - extra_credit

        grades.add_message('%s: %s (%s of %s points)' %
                           ("PASS" if points >= max_points else
                            "FAIL", self.path, regular_credit, max_points))
        if extra_credit > 0:
            grades.add_message('EXTRA CREDIT: %s points' % (extra_credit, ))

        for line in self.messages:
            grades.add_message('    %s' % (line, ))

        return True

    def add_message(self, message):
        """Add generic message."""
        self.messages.extend(message.split('\n'))
