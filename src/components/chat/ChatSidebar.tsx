import type { FormEvent } from 'react';

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
}: ChatSidebarProps) {
  return (
    <div className="chat-sidebar">
      <div className="sidebar-header">
        <h3>My Courses</h3>
        <button onClick={() => onSetCreatingCourse(true)} className="add-button" title="Add Course">+</button>
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

      <div className="sidebar-content">
        {courses.map((course) => (
          <div key={course.id} className="course-group">
            <div
              className={`course-item ${expandedCourseId === course.id ? 'expanded' : ''}`}
              onClick={() => onToggleCourse(course.id)}
            >
              <span className="name">{course.name}</span>
              <span className="dropdown-arrow">›</span>
            </div>

            {expandedCourseId === course.id && (
              <div className="session-list">
                <div className="session-header">
                  <small>Sessions</small>
                  <button onClick={() => onSetCreatingSession(true)} className="add-button-small">+</button>
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
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 24 24"
                          className="folder-icon"
                        >
                          <path d="M19,3H12.472a1.019,1.019,0,0,1-.447-.1L8.869,1.316A3.014,3.014,0,0,0,7.528,1H5A5.006,5.006,0,0,0,0,6V18a5.006,5.006,0,0,0,5,5H19a5.006,5.006,0,0,0,5-5V8A5.006,5.006,0,0,0,19,3ZM5,3H7.528a1.019,1.019,0,0,1,.447.1l3.156,1.579A3.014,3.014,0,0,0,12.472,5H19a3,3,0,0,1,2.779,1.882L2,6.994V6A3,3,0,0,1,5,3ZM19,21H5a3,3,0,0,1-3-3V8.994l20-.113V18A3,3,0,0,1,19,21Z" />
                        </svg>
                        <span className="name">{session.name}</span>
                      </div>
                      <span className="dropdown-arrow">›</span>
                    </div>

                    {expandedSessionId === session.id && (
                      <div className="chat-list">
                        <button onClick={onCreateChat} className="new-chat-button-small">+ New Chat</button>
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
                                  >
                                    ✎
                                  </button>
                                  <button onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteChat(chat.id);
                                  }}
                                  className="action-btn"
                                  >
                                    ×
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
