import { useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
  ChevronsUpDown,
  ChevronRight,
  FolderClosed,
  MessageSquarePlus,
  PanelLeftClose,
  Pencil,
  Plus,
  Search,
  Trash2,
  X,
} from 'lucide-react';

type CourseLike = {
  id: string;
  name: string;
};

type SessionLike = {
  id: string;
  name: string;
};

type ChatLike = {
  id: string;
  name: string;
};

type ChatSidebarProps = {
  courses: CourseLike[];
  sessions: SessionLike[];
  chats: ChatLike[];
  expandedCourseId: string | null;
  expandedSessionId: string | null;
  selectedChatId: string | null;
  isCreatingCourse: boolean;
  newCourseName: string;
  isCreatingSession: boolean;
  newSessionName: string;
  editingChatId: string | null;
  editChatName: string;
  onSetCreatingCourse: (value: boolean) => void;
  onSetNewCourseName: (value: string) => void;
  onCreateCourse: (e: FormEvent) => void;
  onToggleCourse: (courseId: string) => void;
  onSetCreatingSession: (value: boolean) => void;
  onSetNewSessionName: (value: string) => void;
  onCreateSession: (e: FormEvent) => void;
  onToggleSession: (sessionId: string) => void;
  onCreateChat: () => void;
  onSelectChat: (chatId: string) => void;
  onSetEditingChatId: (chatId: string | null) => void;
  onSetEditChatName: (value: string) => void;
  onUpdateChatName: (chatId: string, newName: string) => void;
  onDeleteChat: (chatId: string) => void;
  onCloseSidebar: () => void;
};

export default function ChatSidebar({
  courses,
  sessions,
  chats,
  expandedCourseId,
  expandedSessionId,
  selectedChatId,
  isCreatingCourse,
  newCourseName,
  isCreatingSession,
  newSessionName,
  editingChatId,
  editChatName,
  onSetCreatingCourse,
  onSetNewCourseName,
  onCreateCourse,
  onToggleCourse,
  onSetCreatingSession,
  onSetNewSessionName,
  onCreateSession,
  onToggleSession,
  onCreateChat,
  onSelectChat,
  onSetEditingChatId,
  onSetEditChatName,
  onUpdateChatName,
  onDeleteChat,
  onCloseSidebar,
}: ChatSidebarProps) {
  const [courseQuery, setCourseQuery] = useState('');
  const [lastOpenedCourseId, setLastOpenedCourseId] = useState<string | null>(null);
  const normalizedQuery = courseQuery.trim().toLowerCase();

  const visibleCourses = useMemo(() => {
    const ordered = [...courses];
    if (lastOpenedCourseId) {
      ordered.sort((a, b) => {
        if (a.id === lastOpenedCourseId) return -1;
        if (b.id === lastOpenedCourseId) return 1;
        return a.name.localeCompare(b.name);
      });
    }
    if (!normalizedQuery) return ordered;
    return ordered.filter((course) => course.name.toLowerCase().includes(normalizedQuery));
  }, [courses, lastOpenedCourseId, normalizedQuery]);

  const hasExpandedRows = expandedCourseId !== null || expandedSessionId !== null;

  const handleToggleCourse = (courseId: string) => {
    setLastOpenedCourseId(courseId);
    onToggleCourse(courseId);
  };

  return (
    <div className="chat-sidebar">
      <div className="sidebar-content">
        <div className="sidebar-header">
          <div>
            <h3>My Courses</h3>
            <p className="sidebar-caption">
              {visibleCourses.length} of {courses.length} courses
            </p>
          </div>
          <div className="sidebar-header-actions">
            <button
              onClick={() => onSetCreatingCourse(true)}
              className="add-button"
              title="Add Course"
              type="button"
              aria-label="Add course"
            >
              <Plus size={14} aria-hidden="true" />
            </button>
            <button
              onClick={onCloseSidebar}
              className="add-button"
              title="Close sidebar"
              type="button"
              aria-label="Close sidebar"
            >
              <PanelLeftClose size={14} aria-hidden="true" />
            </button>
          </div>
        </div>

        <div className="sidebar-search-row">
          <div className="sidebar-search">
            <Search size={14} aria-hidden="true" className="sidebar-search-icon" />
            <input
              type="text"
              value={courseQuery}
              onChange={(event) => setCourseQuery(event.target.value)}
              className="sidebar-search-input"
              placeholder="Search courses..."
              aria-label="Search courses"
            />
            {courseQuery.length > 0 && (
              <button
                type="button"
                className="sidebar-search-clear"
                onClick={() => setCourseQuery('')}
                aria-label="Clear search"
                title="Clear search"
              >
                <X size={12} aria-hidden="true" />
              </button>
            )}
          </div>
          <button
            type="button"
            className="sidebar-collapse-btn"
            onClick={() => {
              if (expandedSessionId) onToggleSession(expandedSessionId);
              if (expandedCourseId) onToggleCourse(expandedCourseId);
            }}
            disabled={!hasExpandedRows}
            title="Collapse all"
          >
            <ChevronsUpDown size={14} aria-hidden="true" />
            <span>Collapse</span>
          </button>
        </div>

        {isCreatingCourse && (
          <form onSubmit={onCreateCourse} className="create-form">
            <input
              autoFocus
              value={newCourseName}
              onChange={(e) => onSetNewCourseName(e.target.value)}
              placeholder="Course Name"
              onBlur={() => onSetCreatingCourse(false)}
            />
          </form>
        )}

        {visibleCourses.length === 0 && (
          <div className="empty-state-small">No courses match your search</div>
        )}

        {visibleCourses.map((course) => (
          <div key={course.id} className="course-group">
            <div
              className={`course-item ${expandedCourseId === course.id ? 'expanded' : ''}`}
              onClick={() => handleToggleCourse(course.id)}
            >
              <span className="name">{course.name}</span>
              <span className="dropdown-arrow" aria-hidden="true">
                <ChevronRight size={16} />
              </span>
            </div>

            {expandedCourseId === course.id && (
              <div className="session-list">
                <div className="session-header">
                  <small>Sessions</small>
                  <button
                    onClick={() => onSetCreatingSession(true)}
                    className="add-button-small"
                    type="button"
                    aria-label="Add session"
                  >
                    <Plus size={12} aria-hidden="true" />
                  </button>
                </div>

                {isCreatingSession && (
                  <form onSubmit={onCreateSession} className="create-form-small">
                    <input
                      autoFocus
                      value={newSessionName}
                      onChange={(e) => onSetNewSessionName(e.target.value)}
                      placeholder="Session Name"
                      onBlur={() => onSetCreatingSession(false)}
                    />
                  </form>
                )}

                {sessions.map((session) => (
                  <div key={session.id} className="session-group">
                    <div
                      className={`session-item ${expandedSessionId === session.id ? 'expanded' : ''}`}
                      onClick={() => onToggleSession(session.id)}
                    >
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <FolderClosed className="folder-icon" size={15} aria-hidden="true" />
                        <span className="name">{session.name}</span>
                      </div>
                      <span className="dropdown-arrow" aria-hidden="true">
                        <ChevronRight size={16} />
                      </span>
                    </div>

                    {expandedSessionId === session.id && (
                      <div className="chat-list">
                        <button onClick={onCreateChat} className="new-chat-button-small" type="button">
                          <MessageSquarePlus size={14} aria-hidden="true" />
                          <span>New Chat</span>
                        </button>
                        {chats.map((chat) => (
                          <div
                            key={chat.id}
                            className={`chat-item ${selectedChatId === chat.id ? 'active' : ''}`}
                            onClick={() => onSelectChat(chat.id)}
                          >
                            {editingChatId === chat.id ? (
                              <input
                                value={editChatName}
                                onChange={(e) => onSetEditChatName(e.target.value)}
                                onBlur={() => onUpdateChatName(chat.id, editChatName)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') onUpdateChatName(chat.id, editChatName);
                                  if (e.key === 'Escape') onSetEditingChatId(null);
                                }}
                                autoFocus
                                onClick={(e) => e.stopPropagation()}
                                className="chat-name-input"
                              />
                            ) : (
                              <>
                                <span className="chat-name" onDoubleClick={(e) => {
                                  e.stopPropagation();
                                  onSetEditingChatId(chat.id);
                                  onSetEditChatName(chat.name);
                                }}
                                >
                                  {chat.name}
                                </span>
                                <div className="chat-actions">
                                  <button onClick={(e) => {
                                    e.stopPropagation();
                                    onSetEditingChatId(chat.id);
                                    onSetEditChatName(chat.name);
                                  }}
                                  className="action-btn"
                                  type="button"
                                  aria-label={`Rename ${chat.name}`}
                                  >
                                    <Pencil size={14} aria-hidden="true" />
                                  </button>
                                  <button onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteChat(chat.id);
                                  }}
                                  className="action-btn"
                                  type="button"
                                  aria-label={`Delete ${chat.name}`}
                                  >
                                    <Trash2 size={14} aria-hidden="true" />
                                  </button>
                                </div>
                              </>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
                {sessions.length === 0 && !isCreatingSession && (
                  <div className="empty-state-small">No sessions yet</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
