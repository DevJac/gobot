import fileinput
import re
import sys


def name(args):
    return 'Gobot'


def main():
    command_re = re.compile(r'(?P<id>[0-9]+)? ?(?P<command_name>\w+) ?(?P<args>.*)')
    for line in fileinput.input():
        try:
            command = re.match(command_re, line)
            response = globals()[command.group('command_name')](command.group('args'))
            print(f"={command.group('id') or ''} {response}\n\n")
        except Exception:
            print(f'Failed on command: {repr(line)}', file=sys.stderr)
            try:
                print(f'Parsed command as: {command.groups()}', file=sys.stderr)
            except Exception:
                print("Couldn't parse command", file=sys.stderr)
                pass
            raise


if __name__ == '__main__':
    main()
